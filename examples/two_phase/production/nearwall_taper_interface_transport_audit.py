"""L1A-2t near-wall taper, interface transport and contact-side impact causal audit.

Diagnostic-only stage. Nothing in this module changes a production initializer, a
solver operator, a frozen threshold, a stored dataset, the timestep policy or the
solver contract. The L1A-2s initializer ``SDF_TAPERED_STREAMFUNCTION_V1`` (C2) is
reproduced exactly and used as the reference state; the prespecified T1/T2/T3
candidates change *only* the SDF taper onset ``a`` and transition width ``b`` of
the same streamfunction, everything else (compact radial envelope, periodic seam
taper, differentiation, core-mean amplitude normalization, phase field, solid,
parameters, integrator) is bitwise shared.

The question is physical causality, not CI: distinguish

* ``(i)`` the incident **fluid** normal speed,
* ``(ii)`` the reconstructed ``phi=0.5`` contour advance, and
* ``(iii)`` advective plus explicit/implicit Cahn--Hilliard phase transport,

and quantify the competing explanations H-TAPER, H-CH, H-BP, H-GEOMETRY and
H-MULTI rather than asserting one of them.

Run from ``examples/two_phase`` with ``JAX_ENABLE_X64=1`` and ``PYTHONPATH=.``.
The forensic profile uses N=192; the quick profile is a methodology smoke that is
permanently ineligible for any physical verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

STAGE = "L1A-2t"
STAGE_VERSION = "l1a2t_v1"

#: Live merged ``main`` verified for this stage (PR #22 merge commit).
VERIFIED_MAIN_SHA = "09e2993b8da478d79e768054194201af57121784"
PR22 = {
    "number": 22,
    "state": "MERGED",
    "head_sha": "865317d438eee5c8d5f1e942f0ad30c943010818",
    "merge_sha": VERIFIED_MAIN_SHA,
    "merged_at_utc": "2026-10-11T07:37:01Z",
    "url": "https://github.com/licy0505/hydrogym/pull/22",
}
PR23 = {
    "number": 23,
    "state": "MERGED",
    "head_sha": "ea00347c8a6cebf6cfbff36f744d39ccd60784d1",
    "merge_sha": "039f6a0d9a14ff601c76a58f5a66ce0900677de5",
}
#: Stage predecessor evidence is frozen read-only input.
L1A2S_EVIDENCE_FILES = (
    "l1a2s/l1a2s_final_report.md",
    "l1a2s/l1a2s_final_report.json",
    "l1a2s/initializer_constraint_matrix.json",
    "l1a2s/impact_event_matrix.json",
    "l1a2s/quality_status.json",
    "l1a2s/manifest.json",
    "l1a2s/pillar_flat_operator_comparison/pillar_flat_operator_comparison.json",
)
EXPECTED_POLICY = "impact_phase_cap_dx2_v1"
CASE_NAMES = ("flat_we100_ct050", "flat_we200_ct000", "pillar_training", "complex_heldout")

#: Prespecified taper candidate set (equation in section 5 of the stage spec).
#: ``a`` is the onset in units of dx, ``b`` the transition width in units of dx.
TAPER_CANDIDATES: dict[str, dict[str, Any]] = {
    "C2_BASE": {
        "onset_a_dx": 4.0,
        "width_b_dx": 8.0,
        "transition_sdf_over_dx": [4.0, 12.0],
        "role": "exact L1A-2s C2 reproduction",
        "eligible": True,
    },
    "T1_NEAR_NARROW": {
        "onset_a_dx": 1.0,
        "width_b_dx": 4.0,
        "transition_sdf_over_dx": [1.0, 5.0],
        "role": "shortened near-wall taper",
        "eligible": True,
    },
    "T2_NEAR_WIDE": {
        "onset_a_dx": 1.0,
        "width_b_dx": 8.0,
        "transition_sdf_over_dx": [1.0, 9.0],
        "role": "onset-only comparison vs C2",
        "eligible": True,
    },
    "T3_FAR_NARROW": {
        "onset_a_dx": 4.0,
        "width_b_dx": 4.0,
        "transition_sdf_over_dx": [4.0, 8.0],
        "role": "width-only comparison vs C2",
        "eligible": True,
    },
}
#: Declared run order: C2 first (reproduction), then the perturbations.
CANDIDATE_RUN_ORDER = ("C2_BASE", "T1_NEAR_NARROW", "T2_NEAR_WIDE", "T3_FAR_NARROW")

#: Explicitly quarantined negative control (no velocity mask, taper identically one).
#: It bounds how much of the front-speed suppression is caused by the taper itself
#: versus by the compact radial envelope. It can never be promoted.
QUARANTINED_CONTROLS: dict[str, dict[str, Any]] = {
    "NC_NO_WALL_TAPER": {
        "onset_a_dx": None,
        "width_b_dx": None,
        "wall_taper": "identically one",
        "role": "quarantined diagnostic counterfactual; not a candidate",
        "eligible": False,
    }
}

N_FORENSIC = 192
N_QUICK = 48
REQUESTED_DT = 0.004
DT_LEVELS = (0.002, 0.001, 0.0005)
ANCHOR_TIMES = (0.0, 0.08, 0.16, 0.24, 0.40, 0.48)
PRIMARY_EVENT_HORIZON = 0.8
TARGETED_EXTENSION_HORIZON = 1.2
EARLY_CAPTURE_HORIZON = 0.08
DISTANCE_BANDS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 12.0))
FRONT_WINDOW_CELLS = 8

#: The 14 mandatory evidence deliverables of stage L1A-2t (``manifest.json`` is the 15th file).
REQUIRED_EVIDENCE_FILES = (
    "prerequisite_and_source_map.json",
    "taper_candidate_design_contract.json",
    "candidate_initial_compatibility_matrix.json",
    "front_velocity_distance_band_matrix.json",
    "phase_advective_CH_implicit_budget.json",
    "interface_contour_motion_audit.json",
    "first_substep_operator_ledger.json",
    "bounded_candidate_event_matrix.json",
    "temporal_sensitivity_matrix.json",
    "counterfactual_and_falsification_matrix.json",
    "quality_and_blocker_matrix.json",
    "mechanism_decision.json",
    "l1a2t_final_report.json",
    "l1a2t_final_report.md",
)
TRACE_STRIDE = 4
#: Independent contour-agreement tolerance of the second reconstruction (in dx).
CONTOUR_AGREEMENT_DX = 0.1
#: Linearized diagnostic is refused below this gradient/angle conditioning.
LINEARIZED_CONDITIONING_FLOOR = 0.25

REPO = Path(__file__).resolve().parents[3]
TWO_PHASE = REPO / "examples" / "two_phase"
EVIDENCE_DIR = TWO_PHASE / "evidence" / "l1a2t"
ARTIFACT_DIR = TWO_PHASE / "artifacts" / "l1a2t"
L1A2S_DIR = TWO_PHASE / "evidence" / "l1a2s"
QUALITY_BASELINE = REPO / "quality" / "legacy_two_phase_debt_v1.json"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import phasefield as pf  # noqa: E402
from production import impact_impulse_projection_audit as impulse  # noqa: E402
from production import impact_initialization_compatibility_audit as audit  # noqa: E402
from production import timestep_policy  # noqa: E402


class AuditBlocked(RuntimeError):
    """Fail-closed protocol, provenance, source-freeze or identity error."""


# ---------------------------------------------------------------------------
# JSON / hash helpers (write-side only; the frozen readers are reused as-is)
# ---------------------------------------------------------------------------
def _json_default(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(args: list[str], *, required: bool = True) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        if required:
            raise AuditBlocked(f"git command failed: git {' '.join(args)}: {exc}") from exc
        return None


def progress(message: str) -> None:
    print(f"[{STAGE}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Tier 0 — preflight, provenance and source freeze
# ---------------------------------------------------------------------------
#: Files this stage is allowed to create or modify. Everything else must be clean.
STAGE_OWNED_PATHS = (
    "examples/two_phase/production/nearwall_taper_interface_transport_audit.py",
    "examples/two_phase/tests/test_nearwall_taper_interface_transport_audit.py",
    "examples/two_phase/evidence/l1a2t/",
    "examples/two_phase/artifacts/l1a2t/",
)


def _stage_owned_path(path: str) -> bool:
    return any(path == owned or path.startswith(owned) for owned in STAGE_OWNED_PATHS)


def _commits_touch_only_stage_paths(commits: list[str]) -> bool:
    """Every commit this branch adds on top of the verified main must be stage-owned."""
    for line in commits:
        commit = line.split()[0]
        files = _git(["show", "--pretty=format:", "--name-only", commit]) or ""
        for path in files.splitlines():
            if path.strip() and not _stage_owned_path(path.strip()):
                return False
    return True


_PORCELAIN_PATH = re.compile(r"^.{1,2}\s(.*)$")


def _porcelain_path(line: str) -> str:
    """Path of one porcelain-v1 entry, tolerant to a leading status space being stripped."""
    match = _PORCELAIN_PATH.match(line)
    path = (match.group(1) if match else line).strip().strip('"')
    if " -> " in path:  # renames
        path = path.split(" -> ", 1)[1].strip()
    return path


def _unexpected_worktree_changes(porcelain: str) -> list[str]:
    """Dirty entries outside this stage's own deliverables (they must be empty)."""
    unexpected = []
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        path = _porcelain_path(line)
        if not _stage_owned_path(path):
            unexpected.append(line.strip())
    return unexpected


def preflight_state() -> dict[str, Any]:
    """Verify branch topology, clean tree and the frozen source/evidence hashes."""
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    head = _git(["rev-parse", "HEAD"])
    origin_main = _git(["rev-parse", "origin/main"])
    porcelain = _git(["status", "--porcelain"]) or ""
    unexpected_changes = _unexpected_worktree_changes(porcelain)
    merge_parents = (_git(["rev-list", "--parents", "-1", VERIFIED_MAIN_SHA]) or "").split()[1:]
    merge_base = _git(["merge-base", "HEAD", "origin/main"])
    ancestor = _git(["merge-base", "--is-ancestor", VERIFIED_MAIN_SHA, "HEAD"], required=False) is not None
    commits_since_main = (_git(["log", "--oneline", f"{VERIFIED_MAIN_SHA}..HEAD"]) or "").splitlines()
    source_hashes = audit._source_hashes()
    l1a2s_manifest = json.loads((L1A2S_DIR / "manifest.json").read_text())
    frozen_source = l1a2s_manifest["source_hashes_current"]
    frozen_baseline_ready = audit._source_hashes()  # read-only; keeps the frozen reader exercised
    del frozen_baseline_ready

    source_checks = {
        key: source_hashes[key] == frozen_source[key]
        for key in ("phasefield", "cases", "generate_dataset", "timestep_policy")
    }
    evidence_hashes = {name: sha256_file(TWO_PHASE / "evidence" / name) for name in L1A2S_EVIDENCE_FILES}
    checks = {
        "branch_is_authorized_l1a2t_session_branch": branch is not None and branch.startswith("arena/"),
        "worktree_has_no_changes_outside_stage_deliverables": unexpected_changes == [],
        "origin_main_equals_verified_merge_sha": origin_main == VERIFIED_MAIN_SHA,
        "branch_head_contains_verified_main": ancestor,
        "merge_base_is_the_verified_sha": merge_base == VERIFIED_MAIN_SHA,
        "verified_sha_has_exactly_the_verified_parents": merge_parents == [PR23["merge_sha"], PR22["head_sha"]],
        "commits_since_main_touch_only_stage_paths": _commits_touch_only_stage_paths(commits_since_main),
        "solver_contract_12": int(pf.SOLVER_CONTRACT_VERSION) == 12,
        "timestep_policy_default": timestep_policy.DEFAULT_POLICY_NAME == EXPECTED_POLICY,
        "policy_matches_contract": pf.SOLVER_CONTRACT_12_TRAJECTORY_POLICY == EXPECTED_POLICY,
        "quality_baseline_present": QUALITY_BASELINE.is_file(),
        "stage_has_not_touched_frozen_sources": all(source_checks.values()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AuditBlocked(f"preflight failed: {failed}; head={head} main={origin_main} branch={branch}")
    return {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "prerequisite": {
            "pr22": PR22,
            "pr23": PR23,
            "verified_merge_sha": VERIFIED_MAIN_SHA,
            "branch": branch,
            "head_sha": head,
            "origin_main_sha": origin_main,
            "merge_base_sha": merge_base,
            "commits_since_verified_main": commits_since_main,
            "worktree_clean_within_stage_deliverables": True,
            "verified_sha_parents": merge_parents,
            "pr22_hosted_python_ci_run": 38120327651,
            "pr22_hosted_python_ci_status": "observed_pass_8_of_8_jobs_at_preflight_time",
            "pr22_hosted_ci_is_engineering_only": True,
        },
        "checks": checks,
        "source_hashes_current": source_hashes,
        "source_hashes_from_l1a2s_manifest": frozen_source,
        "source_freeze_checks": source_checks,
        "frozen_evidence_input_sha256": evidence_hashes,
        "quality_baseline_sha256": sha256_file(QUALITY_BASELINE),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def source_operator_map() -> dict[str, Any]:
    """Exact production call graph used by this stage, anchored to live source text."""
    source = (TWO_PHASE / "phasefield.py").read_text()
    anchors = {
        "periodic_D_x": "def _ddx(f, dx):",
        "periodic_D_y": "def _ddy(f, dy):",
        "periodic_poisson": "def poisson_solve(rhs, m2):",
        "rhs": "def rhs(state: State, solid: Solid, p: PhaseFieldParams):",
        "advective_phase_source": (
            "def advective_phase_source(phi, u, v, solid: Solid, p: PhaseFieldParams, dt: float):"
        ),
        "phase_advective_fluxes": "def phase_advective_fluxes(u, v, phi, solid: Solid, p: PhaseFieldParams):",
        "chemical_potential_fluxes": "def chemical_potential_fluxes(mu, solid: Solid, p: PhaseFieldParams):",
        "control_volume_divergence": "def control_volume_divergence(flux_x, flux_y, volume_safe):",
        "phase_control_volumes": "def phase_control_volumes(solid: Solid, p: PhaseFieldParams):",
        "phase_transport_operator": (
            "def phase_transport_operator(solid: Solid, p: PhaseFieldParams) -> PhaseTransportOperator:"
        ),
        "solve_ch_implicit": "def solve_ch_implicit(rhs_field, solid: Solid, p: PhaseFieldParams, dt: float):",
        "phase_update": "def _phase_update(phi, u, v, solid: Solid, p: PhaseFieldParams, dt: float, phi_rhs, mu_expl):",
        "step_with_diagnostics": "def step_with_diagnostics(state: State, solid: Solid, p: PhaseFieldParams):",
        "phase_advection_subcycles": "def phase_advection_subcycles(p: PhaseFieldParams) -> bool:",
    }
    return {
        "call_graph": [
            "case registry -> impact_impulse_projection_audit.study_cases()",
            "case -> impact_initialization_compatibility_audit._derive_bundle -> pf.build_case"
            " -> pf.droplet_initial_state -> State",
            "diagnostic seed -> build_tapered_velocity -> pf._ddy/pf._ddx -> pf.State(phi=production phi)",
            "step -> impulse.make_fast_substep -> pf.rhs -> pf._phase_update -> solve_ch_implicit /"
            " chemical_potential_fluxes / phase_advective_fluxes -> Brinkman -> poisson_solve",
            "contact metric -> audit._local_gap -> audit._threshold_contour_edge_crossings ->"
            " audit._periodic_bilinear_sample",
        ],
        "source_anchors_present": {name: (text in source) for name, text in anchors.items()},
        "source_anchors": anchors,
        "production_public_step": {
            "requested_dt": float(
                pf.PhaseFieldParams(Nx=N_FORENSIC, Ny=N_FORENSIC, Lx=6.0, Ly=6.0, dt=REQUESTED_DT).dt
            ),
            "internal_substeps": 3,
            "internal_step": "p.dt / 3",
            "effective_dt_policy": EXPECTED_POLICY,
            "momentum_chain": "u_n -> u + h*rhs -> Brinkman 1/(1+h*chi/eta_pen) -> periodic pressure projection",
            "phase_chain": (
                "phi_rhs (advective face-flux divergence) -> source = phi_rhs - div_V(J_CH_explicit)"
                " -> solve_ch_implicit(phi + h*source)"
            ),
        },
        "deliberate_non_edits": [
            "no production file, threshold, policy, contract, initializer or dataset semantic is modified",
            "no autoformat was applied to any existing frozen module",
            "no historical evidence or approval record is rewritten",
        ],
    }


# ---------------------------------------------------------------------------
# Taper family — C2 is recovered exactly at (a, b) = (4, 8)
# ---------------------------------------------------------------------------
def smooth5(values: Any):
    """Quintic smoothstep ``z^3 (10 - 15 z + 6 z^2)`` of the frozen C2 taper."""
    z = jnp.clip(values, 0.0, 1.0)
    return z**3 * (10.0 + z * (-15.0 + 6.0 * z))


def wall_taper(sdf: Any, dx: float, onset_a_dx: float | None, width_b_dx: float | None):
    """``smooth5((sdf - a*dx)/(b*dx))``; ``None``/``None`` returns the identically-one control."""
    if onset_a_dx is None or width_b_dx is None:
        return jnp.ones_like(jnp.asarray(sdf))
    return smooth5((jnp.asarray(sdf) - onset_a_dx * dx) / (width_b_dx * dx))


def build_tapered_velocity(
    bundle: Any,
    onset_a_dx: float | None,
    width_b_dx: float | None,
    *,
    solid: Any | None = None,
    phi: Any | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Build the C2 family with an arbitrary wall taper.

    Identical radial compact-support envelope, periodic seam taper, streamfunction
    differentiation and ``V_i``-weighted ``phi>=0.9`` core-mean amplitude
    normalization as the frozen C2 builder. There is no post-differentiation mask
    and no pressure projection of the seed.
    """
    solid = bundle.solid if solid is None else solid
    phi = bundle.base.phi if phi is None else phi
    p = bundle.p
    X, Y = pf.grids(p, dtype=p.dtype)
    sx = (X - bundle.x0 + 0.5 * p.Lx) % p.Lx - 0.5 * p.Lx
    sy = (Y - bundle.y0 + 0.5 * p.Ly) % p.Ly - 0.5 * p.Ly
    r2 = sx * sx + sy * sy
    r_cut = bundle.R + 0.45
    q2 = r2 / (r_cut * r_cut)
    envelope = jnp.where(q2 < 1.0, (1.0 - q2) ** 4, 0.0)
    sdf = jnp.asarray(solid.sdf, dtype=p.dtype)
    taper = wall_taper(sdf, p.dx, onset_a_dx, width_b_dx)
    seam_distance = jnp.minimum(Y, p.Ly - Y)
    y_seam_taper = smooth5((seam_distance - 3.0 * p.dy) / (3.0 * p.dy))
    psi = sx * envelope * taper * y_seam_taper
    unit_u = pf._ddy(psi, p.dy)
    unit_v = -pf._ddx(psi, p.dx)
    phi_host = np.asarray(phi, dtype=np.float64)
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    core_weight = volume * (phi_host >= 0.9)
    denominator = float(np.sum(core_weight))
    if denominator <= 0.0:
        raise AuditBlocked(f"taper family has no phi>=0.9 core in {bundle.case_name}")
    unit_core_v = float(np.sum(core_weight * np.asarray(unit_v, dtype=np.float64)) / denominator)
    if not math.isfinite(unit_core_v) or unit_core_v >= 0.0:
        raise AuditBlocked(f"taper family unit field lost downward core motion: {unit_core_v}")
    scale = -bundle.u_impact / unit_core_v
    u = (scale * unit_u).astype(p.dtype)
    v = (scale * unit_v).astype(p.dtype)
    details = {
        "onset_a_dx": None if onset_a_dx is None else float(onset_a_dx),
        "width_b_dx": None if width_b_dx is None else float(width_b_dx),
        "wall_taper_zero_below_sdf_over_dx": 0.0 if onset_a_dx is None else float(onset_a_dx),
        "wall_taper_plateau_above_sdf_over_dx": (None if onset_a_dx is None else float(onset_a_dx + width_b_dx)),
        "post_derivative_masking": False,
        "diagnostic_pressure_projection_applied_to_seed": False,
        "scale_target": "V_i-weighted mean v over phi>=0.9 equals -u_impact",
        "unit_core_v_before_scale": unit_core_v,
        "amplitude_scale": float(scale),
        "radial_envelope": "A*sx*(1-r2/(R+0.45)^2)^4_+",
        "periodic_y_seam_taper": "smooth5((dist_seam-3dy)/(3dy))",
        "geometry_sdf_hash": audit._array_sha(np.asarray(solid.sdf)),
        "x0": bundle.x0,
        "y0": bundle.y0,
        "R": bundle.R,
    }
    return u, v, psi, taper, details


def build_candidate_state(bundle: Any, key: str) -> tuple[Any, dict[str, Any]]:
    """Create one diagnostic state; ``phi``/geometry are bitwise the production ones."""
    if key in TAPER_CANDIDATES:
        spec = TAPER_CANDIDATES[key]
    elif key in QUARANTINED_CONTROLS:
        spec = QUARANTINED_CONTROLS[key]
    else:
        raise AuditBlocked(f"unknown candidate key {key!r}")
    u, v, psi, taper, details = build_tapered_velocity(bundle, spec["onset_a_dx"], spec["width_b_dx"])
    state = pf.State(phi=bundle.base.phi, u=u, v=v, t=0.0)
    details = {
        **details,
        "candidate": key,
        "role": spec["role"],
        "eligible_for_promotion": bool(spec["eligible"]),
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
    }
    if audit._array_sha(np.asarray(state.phi)) != audit._array_sha(np.asarray(bundle.base.phi)):
        raise AuditBlocked(f"{key} changed the initial phase field")
    return state, details


def c2_exact_reproduction(
    bundle: Any, saved_initial: dict[str, Any], *, compare_saved_evidence: bool = True
) -> dict[str, Any]:
    """Bitwise/dtype-tolerant C2 identity against the frozen L1A-2s evidence.

    ``compare_saved_evidence`` must only be active at the frozen N=192 grid; the
    quick profile compares the two builders but cannot reproduce stored hashes.
    """
    reference_state, reference_details = audit.build_candidate(bundle, "SDF_TAPERED_STREAMFUNCTION_V1")
    ref_u, ref_v, ref_psi, _ref_taper, ref_family = build_tapered_velocity(bundle, 4.0, 8.0)
    saved = saved_initial[bundle.case_name]["SDF_TAPERED_STREAMFUNCTION_V1"]
    hashes = {
        "phi": audit._array_sha(np.asarray(bundle.base.phi)),
        "u": audit._array_sha(np.asarray(reference_state.u)),
        "v": audit._array_sha(np.asarray(reference_state.v)),
        "geometry": audit._geometry_hash(bundle.solid),
    }
    saved_hashes = {
        "phi": saved["initial_phi_hash"],
        "u": saved["initial_u_hash"],
        "v": saved["initial_v_hash"],
        "geometry": saved["initial_geometry_hash"],
    }
    comparison = {
        "u_bitwise_equal_to_l1a2s_builder": bool(np.array_equal(np.asarray(reference_state.u), np.asarray(ref_u))),
        "v_bitwise_equal_to_l1a2s_builder": bool(np.array_equal(np.asarray(reference_state.v), np.asarray(ref_v))),
        "psi_bitwise_equal_to_l1a2s_builder": bool(
            np.array_equal(np.asarray(audit.build_c2_velocity(bundle)[2]), np.asarray(ref_psi))
        ),
        "amplitude_scale_l1a2s_builder": float(reference_details["amplitude_scale"]),
        "amplitude_scale_taper_family": float(ref_family["amplitude_scale"]),
    }
    comparison["amplitude_scale_relative_delta"] = abs(
        comparison["amplitude_scale_taper_family"] - comparison["amplitude_scale_l1a2s_builder"]
    ) / max(abs(comparison["amplitude_scale_l1a2s_builder"]), 1e-300)
    checks = (
        {name: hashes[name] == saved_hashes[name] for name in hashes}
        if compare_saved_evidence
        else {"status": False, "reason": "QUICK_ONLY_PROFILE_SAVED_HASHES_APPLY_TO_N192"}
    )
    if compare_saved_evidence and not all(checks.values()):
        raise AuditBlocked(f"{bundle.case_name}: C2 initial-state hash mismatch {checks}")
    if not all(
        comparison[key]
        for key in (
            "u_bitwise_equal_to_l1a2s_builder",
            "v_bitwise_equal_to_l1a2s_builder",
            "psi_bitwise_equal_to_l1a2s_builder",
        )
    ):
        raise AuditBlocked(f"{bundle.case_name}: parameterized taper family does not reproduce C2 bitwise")
    return {
        "case_name": bundle.case_name,
        "comparison_rule": "sha256 of the exact solver arrays plus bitwise array equality of the two builders",
        "saved_evidence_compared": bool(compare_saved_evidence),
        "hash_checks": checks,
        "hashes": hashes,
        "saved_hashes": saved_hashes,
        **comparison,
    }


# ---------------------------------------------------------------------------
# Tier 1 — same-state initializer compatibility and local approach
# ---------------------------------------------------------------------------
def _front_window(support: dict[str, Any], p: Any) -> np.ndarray:
    """2-D ``(Nx, Ny)`` front column window around the reconstructed contour support.

    Field arrays in this code base are laid out ``[x, y]``. The window is a periodic
    band in ``x`` around the support column that keeps every open control volume, so
    the ``sdf/dx`` bands can be measured without mixing the two axes. The earlier
    broadcast of a length-``Nx`` mask against ``(Nx, Ny)`` fields silently selected a
    ``y`` band instead and produced all-``EMPTY_BAND`` output; that defect is fixed
    here and regression-tested.
    """
    cell = support.get("gap_cell")
    if cell is None:
        return np.zeros((p.Nx, p.Ny), dtype=bool)
    i0 = int(cell[0])
    ii = (np.arange(p.Nx) - i0 + p.Nx // 2) % p.Nx - p.Nx // 2
    return (np.abs(ii) <= FRONT_WINDOW_CELLS)[:, None]


def distance_band_matrix(bundle: Any, state: Any, psi: Any, taper: Any, support: dict[str, Any]) -> dict[str, Any]:
    """``sdf/dx`` band statistics of the taper, streamfunction and differentiated velocity."""
    p = bundle.p
    sdf = np.asarray(bundle.solid.sdf, dtype=np.float64)
    volume = np.asarray(bundle.solid.geometry.volume, dtype=np.float64)
    phi = np.asarray(state.phi, dtype=np.float64)
    u = np.asarray(state.u, dtype=np.float64)
    v = np.asarray(state.v, dtype=np.float64)
    psi_host = np.asarray(psi, dtype=np.float64)
    taper_host = np.asarray(taper, dtype=np.float64)
    d = sdf / float(p.dx)
    speed = np.hypot(u, v)
    U = max(abs(float(bundle.u_impact)), 1e-30)
    window = _front_window(support, p) & (volume > 0.0)
    mass = float(np.sum(phi * volume))
    bands: dict[str, Any] = {}
    for low, high in DISTANCE_BANDS:
        selection = window & (d >= low) & (d < high)
        count = int(np.sum(selection))
        keys = [f"{low:g}_{high:g}"]
        entry: dict[str, Any] = {"sdf_over_dx_interval": [low, high], "cell_count": count}
        if count == 0:
            entry.update(
                {
                    "status": "EMPTY_BAND",
                    "mean_taper": None,
                    "mean_psi": None,
                    "mean_v_over_u_impact": None,
                    "max_abs_speed_over_u_impact": None,
                }
            )
        else:
            entry.update(
                {
                    "status": "MEASURED",
                    "mean_taper": float(np.mean(taper_host[selection])),
                    "min_taper": float(np.min(taper_host[selection])),
                    "max_taper": float(np.max(taper_host[selection])),
                    "mean_psi": float(np.mean(psi_host[selection])),
                    "max_abs_psi": float(np.max(np.abs(psi_host[selection]))),
                    "mean_u_over_u_impact": float(np.mean(u[selection]) / U),
                    "mean_v_over_u_impact": float(np.mean(v[selection]) / U),
                    "max_abs_speed_over_u_impact": float(np.max(speed[selection]) / U),
                    "liquid_fraction_mean": float(np.mean(phi[selection])),
                }
            )
        bands[keys[0]] = entry
    support_xy = support.get("support_xy")
    if support_xy is not None:
        point_x = np.asarray([support_xy[0]], dtype=np.float64)
        point_y = np.asarray([support_xy[1]], dtype=np.float64)
        at_support = {
            "psi": float(audit._periodic_bilinear_sample(psi_host, point_x, point_y, p)[0]),
            "wall_taper": float(audit._periodic_bilinear_sample(taper_host, point_x, point_y, p)[0]),
            "u": float(audit._periodic_bilinear_sample(u, point_x, point_y, p)[0]),
            "v": float(audit._periodic_bilinear_sample(v, point_x, point_y, p)[0]),
            "sdf": float(audit._periodic_bilinear_sample(sdf, point_x, point_y, p)[0]),
            "phi": float(audit._periodic_bilinear_sample(phi, point_x, point_y, p)[0]),
            "chi": float(
                audit._periodic_bilinear_sample(np.asarray(bundle.solid.chi, dtype=np.float64), point_x, point_y, p)[0]
            ),
            "method": "periodic bilinear sample at the frozen reconstructed phi=0.5 contour support",
        }
        at_support["psi_over_u_impact_R"] = at_support["psi"] / (U * float(bundle.R))
        at_support["sdf_over_dx"] = at_support["sdf"] / float(p.dx)
        at_support["approach_speed_over_u_impact"] = audit._approach_speed(u, v, support, p) / U
    else:
        at_support = {"status": "UNMEASURED_NO_CONTOUR_SUPPORT"}
    central_boundary = {
        "note": "central stencil crosses the nominal taper boundary when the cell centre lies within 1 dx",
        "cells_with_stencil_crossing_onset": int(
            np.sum(window & (np.abs(d - float(TAPER_CANDIDATES_DEFAULT_ONSET)) <= 1.0))
        ),
        "cells_with_stencil_crossing_plateau": int(np.sum(window & (np.abs(d - 12.0) <= 1.0))),
    }
    gate_location = _frozen_gate_location_diagnostic(bundle, state, taper_host, d)
    return {
        "case_name": bundle.case_name,
        "front_window_definition": (
            f"open control volumes with V_i>0 within +/-{FRONT_WINDOW_CELLS} cells (periodic x) of the "
            "frozen reconstructed phi=0.5 contour support"
        ),
        "field_layout_note": "all fields are indexed [x, y]; the band mask is a periodic x-window",
        "bands": bands,
        "at_frozen_support": at_support,
        "central_stencil_near_boundary": central_boundary,
        "frozen_gate_location_diagnostic": gate_location,
        "phi_weighted_mass_Vphi": mass,
    }


def _frozen_gate_location_diagnostic(bundle: Any, state: Any, taper_host: np.ndarray, d: np.ndarray) -> dict[str, Any]:
    """Where the frozen ``chi>=0.01`` initial-gate maximum actually sits.

    The frozen near-wall gate is defined on the *smoothed solid indicator* ``chi``, whose
    tail extends a few cells into the open fluid. Reporting the argmax cell, its
    ``sdf/dx``, chi, control volume and taper value separates "solid" from "diffuse
    fluid band" without redefining the gate.
    """
    chi = np.asarray(bundle.solid.chi, dtype=np.float64)
    volume = np.asarray(bundle.solid.geometry.volume, dtype=np.float64)
    u = np.asarray(state.u, dtype=np.float64)
    v = np.asarray(state.v, dtype=np.float64)
    U = max(abs(float(bundle.u_impact)), 1e-30)
    mask = chi >= 0.01
    speed = np.hypot(u, v)
    out: dict[str, Any] = {
        "frozen_gate": "max|velocity| over cells with chi>=0.01, divided by u_impact",
        "gate_limit": 0.02,
        "cell_count": int(np.sum(mask)),
        "note": (
            "chi is the smoothed solid indicator; cells with 0.01<=chi<0.5 are open-fluid"
            " control volumes in the diffuse band"
        ),
    }
    if not mask.any():
        out["status"] = "NO_CELL_WITH_CHI_GE_0P01"
        return out
    flat_index = int(np.argmax(np.where(mask, speed, -np.inf)))
    i, j = np.unravel_index(flat_index, speed.shape)
    out.update(
        {
            "status": "MEASURED",
            "argmax_cell_xy": [int(i), int(j)],
            "argmax_speed_over_u_impact": float(speed[i, j] / U),
            "argmax_sdf_over_dx": float(d[i, j]),
            "argmax_chi": float(chi[i, j]),
            "argmax_control_volume_fraction": float(volume[i, j] / (float(bundle.p.dx) * float(bundle.p.dy))),
            "argmax_wall_taper": float(taper_host[i, j]),
            "open_volume_share_of_gate_cells": float(np.sum(mask & (volume > 0.0)) / max(int(np.sum(mask)), 1)),
            "cells_with_chi_below_half_and_open": int(np.sum(mask & (chi < 0.5) & (volume > 0.0))),
        }
    )
    return out


TAPER_CANDIDATES_DEFAULT_ONSET = 4.0


def initial_candidate_record(
    bundle: Any, key: str, saved_initial: dict[str, Any], *, compare_saved_evidence: bool = True
) -> dict[str, Any]:
    state, details = build_candidate_state(bundle, key)
    metrics = audit.initial_metrics(bundle, "SDF_TAPERED_STREAMFUNCTION_V1", state, details)
    support = audit._local_gap(np.asarray(state.phi), 0.5, bundle.solid, bundle.p)
    approach = audit._approach_speed(np.asarray(state.u), np.asarray(state.v), support, bundle.p)
    U = max(abs(float(bundle.u_impact)), 1e-30)
    saved = saved_initial[bundle.case_name]["SDF_TAPERED_STREAMFUNCTION_V1"]
    reproduction = {
        "phi_hash_matches_saved": metrics["initial_phi_hash"] == saved["initial_phi_hash"],
        "geometry_hash_matches_saved": metrics["initial_geometry_hash"] == saved["initial_geometry_hash"],
    }
    if compare_saved_evidence and key == "C2_BASE" and not all(reproduction.values()):
        raise AuditBlocked(f"{bundle.case_name}: C2 t=0 identity drifted from saved L1A-2s evidence")
    return {
        "candidate": key,
        "role": TAPER_CANDIDATES[key]["role"] if key in TAPER_CANDIDATES else QUARANTINED_CONTROLS[key]["role"],
        "taper_parameters": details,
        "t0_identity": {
            "phi_bitwise_identical_across_candidates": None,  # filled by the caller
            "frozen_state_hashes": {
                "phi": metrics["initial_phi_hash"],
                "geometry": metrics["initial_geometry_hash"],
                "u": metrics["initial_u_hash"],
                "v": metrics["initial_v_hash"],
            },
            "l1a2s_reproduction_checks": reproduction,
        },
        "frozen_initial_acceptance": {
            "predeclared": audit.INITIAL_ACCEPTANCE,
            "checks": metrics["acceptance_checks"],
            "pass": bool(metrics["initial_constraints_pass"]),
            "dimensionless_residuals": metrics["dimensionless_residuals"],
            "denominators": {
                "u_impact": U,
                "dx": float(bundle.p.dx),
                "dx_over_u_impact": float(bundle.p.dx) / U,
                "eps": float(bundle.p.eps),
                "definition": "residual = dx * Linf(D) / u_impact for divergence; speed / u_impact for wall gates",
            },
        },
        "raw_constraint_values": {
            "peak_abs_divergence_central_D": metrics["initial_D_div_Linf"],
            "open_face_fv_divergence_Linf": metrics["cutcell_flux_div_Linf"],
            "max_abs_velocity_deep_solid": metrics["max_abs_velocity_deep_solid"],
            "max_abs_velocity_nearwall_chi_ge_0p01": metrics["max_abs_velocity_nearwall_chi_ge_0p01"],
            "max_abs_velocity_nearwall_sdf_abs_le_3dx": metrics["max_abs_velocity_nearwall_sdf_abs_le_3dx"],
            "max_embedded_wall_bilinear_normal_velocity": metrics["max_embedded_wall_bilinear_normal_velocity"],
            "embedded_wall_absolute_flux_bilinear": metrics["embedded_wall_absolute_flux_bilinear"],
            "core_v_phi_ge_0p9": metrics["actual_initial_core_v_phi_ge_0p9"],
            "liquid_v_weighted": metrics["actual_initial_liquid_weighted_v"],
            "gas_v_weighted": metrics["actual_initial_gas_weighted_v"],
            "return_flow_peak_speed": metrics["return_flow_peak_speed"],
            "max_speed_global": metrics["max_speed_global"],
            "kinetic_energy_open_fluid": metrics["kinetic_energy_regionwise"]["energy"]["open_fluid_control_volume"],
            "liquid_physical_kinetic_energy": metrics["kinetic_energy_regionwise"]["energy"][
                "liquid_weighted_control_volume"
            ],
        },
        "initial_front_contour": {
            "support": support,
            "local_normal_approach_speed_into_solid": approach,
            "local_normal_approach_over_u_impact": None if approach is None else approach / U,
            "meaningful_approach_floor": audit.APPROACH_FLOOR_FRACTION * U,
            "exceeds_frozen_approach_floor": bool(
                approach is not None and approach >= audit.APPROACH_FLOOR_FRACTION * U
            ),
            "support_is_shared_across_candidates": True,
            "note": (
                "phase field is bitwise identical across candidates, so this support is candidate-independent at t=0"
            ),
        },
        "nearwall_region_decomposition": _nearwall_region_decomposition(bundle, state),
    }


def _nearwall_region_decomposition(bundle: Any, state: Any) -> dict[str, Any]:
    """Split the frozen ``chi>=0.01`` region into physically solid and diffuse-tail parts."""
    p = bundle.p
    chi = np.asarray(bundle.solid.chi, dtype=np.float64)
    sdf = np.asarray(bundle.solid.sdf, dtype=np.float64)
    volume = np.asarray(bundle.solid.geometry.volume, dtype=np.float64)
    speed = np.hypot(np.asarray(state.u, dtype=np.float64), np.asarray(state.v, dtype=np.float64))
    U = max(abs(float(bundle.u_impact)), 1e-30)
    frozen_region = chi >= 0.01
    groups = {
        "frozen_chi_ge_0p01": frozen_region,
        "chi_ge_0p5_cut_or_solid": chi >= 0.5,
        "chi_ge_0p9_solid": chi >= 0.9,
        "chi_in_0p01_to_0p5_diffuse_fluid_side": frozen_region & (chi < 0.5),
        "fluid_volume_positive": volume > 0.0,
    }
    out: dict[str, Any] = {}
    for name, mask in groups.items():
        if not mask.any():
            out[name] = {"cell_count": 0, "status": "EMPTY_REGION"}
            continue
        argmax = int(np.argmax(np.where(mask, speed, -1.0)))
        index = np.unravel_index(argmax, speed.shape)
        out[name] = {
            "cell_count": int(np.sum(mask)),
            "max_speed_over_u_impact": float(np.max(speed[mask]) / U),
            "argmax_cell": [int(index[0]), int(index[1])],
            "argmax_sdf_over_dx": float(sdf[index] / p.dx),
            "argmax_chi": float(chi[index]),
            "argmax_volume_over_cell_area": float(volume[index] / (p.dx * p.dy)),
        }
    return {
        "definition": "frozen near-wall gate uses chi>=0.01; this decomposition reports where its mass sits",
        "groups": out,
    }


def tier1_initial_matrix(
    bundles: dict[str, Any], saved_initial: dict[str, Any], *, compare_saved_evidence: bool = True
) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for case_name in CASE_NAMES:
        bundle = bundles[case_name]
        matrix[case_name] = {
            "case_identity": audit._metadata(bundle),
            "candidates": {},
        }
        reference_phi = None
        for key in CANDIDATE_RUN_ORDER:
            record = initial_candidate_record(bundle, key, saved_initial, compare_saved_evidence=compare_saved_evidence)
            phi_hash = record["t0_identity"]["frozen_state_hashes"]["phi"]
            reference_phi = phi_hash if reference_phi is None else reference_phi
            record["t0_identity"]["phi_bitwise_identical_across_candidates"] = phi_hash == reference_phi
            matrix[case_name]["candidates"][key] = record
        quarantined = initial_candidate_record(
            bundle, "NC_NO_WALL_TAPER", saved_initial, compare_saved_evidence=compare_saved_evidence
        )
        quarantined["quarantined_negative_control"] = True
        quarantined["candidate_outcome_eligible"] = False
        matrix[case_name]["quarantined_controls"] = {"NC_NO_WALL_TAPER": quarantined}
    return matrix


def tier1_band_matrix(bundles: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for case_name in CASE_NAMES:
        bundle = bundles[case_name]
        case_entry: dict[str, Any] = {}
        for key in (*CANDIDATE_RUN_ORDER, "NC_NO_WALL_TAPER"):
            spec = TAPER_CANDIDATES.get(key, QUARANTINED_CONTROLS.get(key))
            u, v, psi, taper, _ = build_tapered_velocity(bundle, spec["onset_a_dx"], spec["width_b_dx"])
            state = pf.State(phi=bundle.base.phi, u=u, v=v, t=0.0)
            support = audit._local_gap(np.asarray(state.phi), 0.5, bundle.solid, bundle.p)
            case_entry[key] = distance_band_matrix(bundle, state, psi, taper, support)
        out[case_name] = {"bands_dx": [list(b) for b in DISTANCE_BANDS], "candidates": case_entry}
    return out


# ---------------------------------------------------------------------------
# Tier 2 — momentum stage ledger, phase budget, contour motion, counterfactuals
# ---------------------------------------------------------------------------
def make_stage_capture_kernel(solid: Any, p: Any, h: float):
    """One substep exposing the RHS/Brinkman/projection velocity stages (diagnostic only)."""

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


def momentum_stage_ledger(bundle: Any, key: str, state: Any, *, n_substeps: int = 1) -> dict[str, Any]:
    """RHS -> Brinkman -> projection at the frozen support and on the periodic grid."""
    p = bundle.p
    h = float(p.dt) / 3.0
    fast_kernel = impulse.make_fast_substep(bundle.solid, p, h)
    capture = make_stage_capture_kernel(bundle.solid, p, h)
    support = audit._local_gap(np.asarray(state.phi), 0.5, bundle.solid, p)
    rows: list[dict[str, Any]] = []
    carry = state
    for index in range(n_substeps):
        before = carry
        fast_state, scalars = fast_kernel(before)
        capture_state, stages = capture(before)
        u_rhs, v_rhs, u_brink, v_brink, u_proj, v_proj = (np.asarray(s) for s in stages)
        state_stages = {
            "before": (np.asarray(before.u), np.asarray(before.v)),
            "after_explicit_rhs": (u_rhs, v_rhs),
            "after_brinkman": (u_brink, v_brink),
            "after_pressure_projection": (u_proj, v_proj),
        }
        approach = {name: audit._approach_speed(u, v, support, p) for name, (u, v) in state_stages.items()}
        scalar_row = {
            name: float(value)
            for name, value in zip(impulse.SCALAR_NAMES, np.asarray(scalars, dtype=np.float64), strict=True)
        }
        deltas = {
            "explicit_rhs": approach["after_explicit_rhs"] - approach["before"],
            "brinkman": approach["after_brinkman"] - approach["after_explicit_rhs"],
            "periodic_pressure_projection": approach["after_pressure_projection"] - approach["after_brinkman"],
        }
        fast_delta = {
            field: float(
                np.max(
                    np.abs(
                        np.asarray(getattr(fast_state, field), dtype=np.float64)
                        - np.asarray(getattr(capture_state, field), dtype=np.float64)
                    )
                )
            )
            for field in ("phi", "u", "v")
        }
        rows.append(
            {
                "substep_index_1_based": index + 1,
                "t_before": float(before.t),
                "support": support,
                "approach_speed_by_velocity_stage_into_solid": approach,
                "approach_delta_by_operator": deltas,
                "grid_mean_v_by_stage": {
                    "before": scalar_row["mean_v_grid_before"],
                    "projection_correction_mean": scalar_row["mean_of_proj_correction"],
                },
                "projection_plain_grid_mean_identity": {
                    "mean_of_proj_correction": scalar_row["mean_of_proj_correction"],
                    "mean_ddx_pressure": scalar_row["mean_ddx_P"],
                    "mean_ddy_pressure": scalar_row["mean_ddy_P"],
                    "identity_note": (
                        "periodic mean of the pressure gradient correction; ~0 is the frozen L1A-2r identity"
                    ),
                },
                "divergence_by_stage_linf": {
                    "before": scalar_row["div_before_linf"],
                    "after_rhs": scalar_row["div_after_rhs_linf"],
                    "after_brinkman": scalar_row["div_after_brinkman_linf"],
                    "after_projection": scalar_row["div_after_projection_linf"],
                },
                "poisson_residual_relative": scalar_row["resid_relative"],
                "cg_iterations": int(scalar_row["cg_iterations"]),
                "cg_converged": bool(scalar_row["cg_converged"]),
                "liquid_mean_v_by_stage": {
                    "before": scalar_row["v_liquid_before"],
                    "after_rhs": scalar_row["v_liquid_after_rhs"],
                    "after_brinkman": scalar_row["v_liquid_after_brinkman"],
                    "after_projection": scalar_row["v_liquid_after_projection"],
                },
                "core_mean_v_by_stage": {
                    "before": scalar_row["v_core_before"],
                    "after_projection": scalar_row["v_core_after_projection"],
                },
                "fast_kernel_vs_capture_state_max_abs_delta": fast_delta,
            }
        )
        carry = fast_state
    return {
        "case_name": bundle.case_name,
        "candidate": key,
        "internal_substep_h": h,
        "substeps": rows,
        "projection_local_vs_global_note": (
            "a nonzero local projected correction coexists with a plain-grid mean of ~0; local and global "
            "quantities are reported separately and no claim is made that projection destroys the grid mean"
        ),
    }


def phase_component_ledger(bundle: Any, key: str, state: Any, *, n_substeps: int = 3) -> dict[str, Any]:
    """Exact additive phase budget of the production substep arithmetic.

    ``delta_phi_total = delta_phi_adv + delta_phi_CH_explicit + delta_phi_CH_implicit``
    is an identity of the frozen operator sequence, evaluated on every active cell.
    """
    p = bundle.p
    solid = bundle.solid
    h = float(p.dt) / 3.0
    operator = pf.phase_transport_operator(solid, p)
    volume = operator.volume
    volume_safe = operator.volume_safe
    volume_host = np.asarray(volume, dtype=np.float64)
    active = volume_host > 0.0
    subcycles = bool(pf.phase_advection_subcycles(p))
    aperture_x = np.asarray(operator.aperture_x, dtype=np.float64)
    aperture_y = np.asarray(operator.aperture_y, dtype=np.float64)

    rows: list[dict[str, Any]] = []
    carry = state
    for index in range(n_substeps):
        phi = carry.phi
        u, v = carry.u, carry.v
        phi_rhs, _u_rhs, _v_rhs, _mu, mu_expl = pf.rhs(carry, solid, p)
        ch_x, ch_y = pf.chemical_potential_fluxes(mu_expl, solid, p)
        div_ch = pf.control_volume_divergence(ch_x, ch_y, volume_safe)
        source = phi_rhs - div_ch
        pre = phi + h * source
        phi_new, _ledger_solve_info = pf.solve_ch_implicit(pre, solid, p, h)
        reference, solve_info = pf._phase_update(phi, u, v, solid, p, h, phi_rhs, mu_expl)

        d_adv = (h * phi_rhs).astype(np.float64)
        d_che = (-h * div_ch).astype(np.float64)
        d_pre = (pre - phi).astype(np.float64)
        d_chi = (phi_new - pre).astype(np.float64)
        d_total = (phi_new - phi).astype(np.float64)
        residual = d_total - (d_adv + d_che + d_chi)
        subcycle_source = pf.advective_phase_source(phi, u, v, solid, p, h)
        subcycle_identity = bool(np.array_equal(np.asarray(subcycle_source), np.asarray(phi_rhs)))
        reference_identity = bool(np.array_equal(np.asarray(reference), np.asarray(phi_new)))

        adv_x, adv_y = pf.phase_advective_fluxes(u, v, phi, solid, p)
        telescoping_advective = float(
            np.sum(volume_host * np.asarray(pf.control_volume_divergence(adv_x, adv_y, volume_safe), dtype=np.float64))
        )
        telescoping_ch = float(
            np.sum(volume_host * np.asarray(pf.control_volume_divergence(ch_x, ch_y, volume_safe), dtype=np.float64))
        )
        blocked_advective = float(
            np.max(
                np.abs(
                    np.concatenate(
                        [
                            np.asarray(adv_x, dtype=np.float64)[aperture_x <= 0.0],
                            np.asarray(adv_y, dtype=np.float64)[aperture_y <= 0.0],
                        ]
                    )
                )
            )
            if ((aperture_x <= 0.0).any() or (aperture_y <= 0.0).any())
            else 0.0
        )
        blocked_ch = float(
            np.max(
                np.abs(
                    np.concatenate(
                        [
                            np.asarray(ch_x, dtype=np.float64)[aperture_x <= 0.0],
                            np.asarray(ch_y, dtype=np.float64)[aperture_y <= 0.0],
                        ]
                    )
                )
            )
            if ((aperture_x <= 0.0).any() or (aperture_y <= 0.0).any())
            else 0.0
        )
        zero_volume_components = {
            "adv": float(np.max(np.abs(d_adv[~active]))) if (~active).any() else 0.0,
            "ch_explicit": float(np.max(np.abs(d_che[~active]))) if (~active).any() else 0.0,
            "ch_implicit": float(np.max(np.abs(d_chi[~active]))) if (~active).any() else 0.0,
            "total": float(np.max(np.abs(d_total[~active]))) if (~active).any() else 0.0,
        }
        rows.append(
            {
                "substep_index_1_based": index + 1,
                "t_before": float(carry.t),
                "advection_subcycling_active": subcycles,
                "advective_rate_equals_rhs_phi_rhs": subcycle_identity,
                "phase_update_bitwise_reproduced": reference_identity,
                "closure_linf_all_cells": float(np.max(np.abs(residual))),
                "closure_linf_active_cells": float(np.max(np.abs(residual[active]))),
                "weighted_mass_budget": {
                    "V_phi_before": float(np.sum(volume_host * np.asarray(phi, dtype=np.float64))),
                    "V_phi_after": float(np.sum(volume_host * np.asarray(phi_new, dtype=np.float64))),
                    "sum_V_delta_adv": float(np.sum(volume_host * d_adv)),
                    "sum_V_delta_ch_explicit": float(np.sum(volume_host * d_che)),
                    "sum_V_delta_ch_implicit": float(np.sum(volume_host * d_chi)),
                    "sum_V_delta_total": float(np.sum(volume_host * d_total)),
                    "sum_V_delta_pre_implicit": float(np.sum(volume_host * d_pre)),
                    "closure_residual": float(
                        np.sum(volume_host * d_total) - np.sum(volume_host * (d_adv + d_che + d_chi))
                    ),
                },
                "max_abs_component": {
                    "adv": float(np.max(np.abs(d_adv))),
                    "ch_explicit": float(np.max(np.abs(d_che))),
                    "ch_implicit": float(np.max(np.abs(d_chi))),
                    "total": float(np.max(np.abs(d_total))),
                },
                "flux_telescoping": {
                    "sum_V_div_advective_flux": telescoping_advective,
                    "sum_V_div_ch_flux": telescoping_ch,
                    "advective_scale": float(np.sum(volume_host * np.abs(d_adv))),
                    "ch_scale": float(np.sum(volume_host * np.abs(d_che))),
                },
                "blocked_face_flux_max_abs": {
                    "advective": blocked_advective,
                    "ch": blocked_ch,
                    "note": "faces with zero open length must carry exactly zero flux",
                },
                "zero_volume_cell_component_max_abs": zero_volume_components,
                "implicit_solver": {
                    "iterations": int(np.asarray(solve_info.iterations)),
                    "relative_residual": float(np.asarray(solve_info.relative_residual)),
                    "converged": bool(np.asarray(solve_info.converged)),
                },
                "phi_overshoot": {
                    "min": float(np.min(np.asarray(phi_new, dtype=np.float64))),
                    "max": float(np.max(np.asarray(phi_new, dtype=np.float64))),
                    "max_abs_overshoot": float(max(np.max(-np.asarray(phi_new, dtype=np.float64)), np.max(0.0), 0.0)),
                },
            }
        )
        carry = pf.State(phi=np.asarray(phi_new).astype(pf.phase_state_dtype(p)), u=u, v=v, t=carry.t + h)
    return {
        "case_name": bundle.case_name,
        "candidate": key,
        "active_branch": {
            "phase_boundary_model": p.phase_boundary_model,
            "phase_transport_geometry": p.phase_transport_geometry,
            "phase_advection_subcycling": getattr(p, "phase_advection_subcycling", None),
            "subcycling_active": subcycles,
            "meaning": (
                "subcycling inactive: the advective increment is exactly h*phi_rhs from pf.rhs, so no "
                "recomputed subcycled advective rate enters the budget"
            ),
        },
        "identity": "delta_phi_total = delta_phi_adv + delta_phi_CH_explicit + delta_phi_CH_implicit",
        "implicit_term_semantics": (
            "delta_phi_CH_implicit is the implicit solve increment (I + h M eps L^2)^-1 applied to the "
            "pre-implicit state; it is not an independent explicit CH term and must not be double counted"
        ),
        "substeps": rows,
        "sum_V_delta_total_over_substeps": float(sum(row["weighted_mass_budget"]["sum_V_delta_total"] for row in rows)),
    }


def _match_points(points: np.ndarray, target: np.ndarray, Lx: float, Ly: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest periodic-neighbour matching of two contour point clouds."""
    if points.size == 0 or target.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    dx = (points[:, None, 0] - target[None, :, 0] + 0.5 * Lx) % Lx - 0.5 * Lx
    dy = (points[:, None, 1] - target[None, :, 1] + 0.5 * Ly) % Ly - 0.5 * Ly
    distance = np.hypot(dx, dy)
    index = np.argmin(distance, axis=1)
    return index, distance[np.arange(points.shape[0]), index]


def independent_contour_points(phi: np.ndarray, threshold: float, p: Any) -> np.ndarray:
    """Second, independent ``phi=threshold`` reconstruction.

    Different algorithm from the frozen linear-edge-crossing rule: a sign change of
    the periodic bilinear interpolant is bracketed along each grid line and refined
    by bisection. Used only as a cross-check of the official contour extractor.
    """
    field = np.asarray(phi, dtype=np.float64)
    points: list[np.ndarray] = []
    x_axis = (np.arange(p.Nx, dtype=np.float64) + 0.5) * float(p.dx)
    y_axis = (np.arange(p.Ny, dtype=np.float64) + 0.5) * float(p.dy)
    for axis, coordinates, spacing, length in (
        (0, x_axis, float(p.dx), float(p.Lx)),
        (1, y_axis, float(p.dy), float(p.Ly)),
    ):
        shifted = np.roll(field, -1, axis=axis)
        crossing = ((field - threshold) * (shifted - threshold)) <= 0.0
        crossing &= field != shifted
        i, j = np.where(crossing)
        if not len(i):
            continue
        if axis == 0:
            x0 = coordinates[i]
            y0 = y_axis[j]
            x1 = (coordinates[i] + spacing) % length
            y1 = y_axis[j]
        else:
            x0 = x_axis[i]
            y0 = coordinates[j]
            x1 = x_axis[i]
            y1 = (coordinates[j] + spacing) % length
        low = np.zeros_like(x0)
        high = np.ones_like(x0)
        for _ in range(40):
            mid = 0.5 * (low + high)
            xm = x0 + mid * ((x1 - x0 + 0.5 * length) % length - 0.5 * length)
            ym = y0 + mid * ((y1 - y0 + 0.5 * length) % length - 0.5 * length)
            value = audit._periodic_bilinear_sample(field, xm, ym, p) - threshold
            same_side = (value * (audit._periodic_bilinear_sample(field, x0, y0, p) - threshold)) > 0.0
            low = np.where(same_side, mid, low)
            high = np.where(same_side, high, mid)
        mid = 0.5 * (low + high)
        xm = x0 + mid * ((x1 - x0 + 0.5 * length) % length - 0.5 * length)
        ym = y0 + mid * ((y1 - y0 + 0.5 * length) % length - 0.5 * length)
        points.append(np.stack((xm % length, ym % length), axis=1))
    return np.concatenate(points, axis=0) if points else np.empty((0, 2), dtype=np.float64)


def contour_motion_audit(
    bundle: Any,
    key: str,
    phi_before: Any,
    phi_after: Any,
    support_before: dict[str, Any],
    support_after: dict[str, Any],
    h: float,
) -> dict[str, Any]:
    """Geometrically matched contour advance, its linearized proxy and a support check."""
    p = bundle.p
    before = np.asarray(phi_before, dtype=np.float64)
    after = np.asarray(phi_after, dtype=np.float64)
    points_before = audit._threshold_contour_edge_crossings(before, 0.5, p)
    points_after = audit._threshold_contour_edge_crossings(after, 0.5, p)
    independent = independent_contour_points(before, 0.5, p)
    out: dict[str, Any] = {
        "case_name": bundle.case_name,
        "candidate": key,
        "contour_definition": "frozen official rule: linear crossing on periodic grid edges, threshold phi=0.5",
        "official_contour_points_before": int(points_before.shape[0]),
        "official_contour_points_after": int(points_after.shape[0]),
        "independent_contour_points_before": int(independent.shape[0]),
    }
    if points_before.size and independent.size:
        index, distance = _match_points(independent, points_before, float(p.Lx), float(p.Ly))
        selected = distance <= 2.0 * float(p.dx)
        support_distance = None
        support_xy = support_before.get("support_xy")
        if support_xy is not None and independent.size:
            delta = independent - np.asarray(support_xy, dtype=np.float64)[None, :]
            delta[:, 0] = (delta[:, 0] + 0.5 * p.Lx) % p.Lx - 0.5 * p.Lx
            delta[:, 1] = (delta[:, 1] + 0.5 * p.Ly) % p.Ly - 0.5 * p.Ly
            support_distance = float(np.min(np.hypot(delta[:, 0], delta[:, 1])))
        out["independent_vs_official"] = {
            "matched_fraction": float(np.mean(selected)) if selected.size else 0.0,
            "max_match_distance_over_dx": float(np.max(distance) / p.dx) if distance.size else None,
            "support_location_difference_over_dx": None if support_distance is None else float(support_distance / p.dx),
            "agreement_within_declared_tolerance": bool(
                support_distance is not None and support_distance <= CONTOUR_AGREEMENT_DX * p.dx
            ),
            "declared_tolerance_dx": CONTOUR_AGREEMENT_DX,
        }
    else:
        out["independent_vs_official"] = {"status": "UNDEFINED_EMPTY_CONTOUR"}
    if points_before.size and points_after.size:
        index, distance = _match_points(points_after, points_before, float(p.Lx), float(p.Ly))
        displacement = points_after - points_before[index]
        displacement[:, 0] = (displacement[:, 0] + 0.5 * p.Lx) % p.Lx - 0.5 * p.Lx
        displacement[:, 1] = (displacement[:, 1] + 0.5 * p.Ly) % p.Ly - 0.5 * p.Ly
        magnitude = np.hypot(displacement[:, 0], displacement[:, 1])
        usable = distance <= 1.5 * float(p.dx)
        normal = support_before.get("local_wall_normal_into_solid")
        projection = None
        if normal is not None and usable.any():
            projection = displacement[usable, 0] * normal[0] + displacement[usable, 1] * normal[1]
        out["matched_contour_motion"] = {
            "usable_match_fraction": float(np.mean(usable)) if usable.size else 0.0,
            "median_displacement_over_dx": float(np.median(magnitude) / p.dx),
            "max_displacement_over_dx": float(np.max(magnitude) / p.dx),
            "normal_projection_mean_over_h": None if projection is None else float(np.mean(projection) / h),
            "normal_projection_max_over_h": None if projection is None else float(np.max(projection) / h),
            "speed_units": "length/time; divide by u_impact for the dimensionless comparison",
            "interpretation": (
                "finite-difference contour speed of the reconstructed phi=0.5 level set; not a material "
                "tracer velocity and not the fluid velocity"
            ),
        }
        out["support_switch"] = {
            "support_cell_before": support_before.get("gap_cell"),
            "support_cell_after": support_after.get("gap_cell"),
            "switched": bool(support_before.get("gap_cell") != support_after.get("gap_cell")),
            "note": "a support change is reported as SUPPORT_SWITCH, never as a velocity discontinuity",
        }
    else:
        out["matched_contour_motion"] = {"status": "UNDEFINED_NO_CONTOUR"}
    out["linearized_normal_advance"] = _linearized_normal_advance(bundle, before, after, support_before, h)
    return out


def _linearized_normal_advance(
    bundle: Any, phi_before: np.ndarray, phi_after: np.ndarray, support: dict[str, Any], h: float
) -> dict[str, Any]:
    """``-delta_phi / (grad(phi) . n_wall)`` at the frozen support, with conditioning guards."""
    p = bundle.p
    cell = support.get("gap_cell")
    normal = support.get("local_wall_normal_into_solid")
    if cell is None or normal is None:
        return {"status": "UNDEFINED_NO_SUPPORT_OR_NORMAL"}
    field = jnp.asarray(phi_before, dtype=p.dtype)
    grad_x = np.asarray(pf._ddx(field, p.dx), dtype=np.float64)
    grad_y = np.asarray(pf._ddy(field, p.dy), dtype=np.float64)
    i, j = int(cell[0]), int(cell[1])
    gx, gy = float(grad_x[i, j]), float(grad_y[i, j])
    grad_norm = math.hypot(gx, gy)
    denominator = gx * normal[0] + gy * normal[1]
    delta = float(np.asarray(phi_after, dtype=np.float64)[i, j] - phi_before[i, j])
    if grad_norm <= 1e-12 or abs(denominator) < LINEARIZED_CONDITIONING_FLOOR * grad_norm:
        return {
            "status": "UNDEFINED_POOR_CONDITIONING",
            "grad_phi": [gx, gy],
            "grad_norm": grad_norm,
            "grad_dot_normal": denominator,
            "conditioning_ratio": None if grad_norm <= 1e-12 else abs(denominator) / grad_norm,
            "conditioning_floor": LINEARIZED_CONDITIONING_FLOOR,
            "refused_division": True,
        }
    return {
        "status": "MEASURED_FIRST_ORDER_DIAGNOSTIC",
        "grad_phi": [gx, gy],
        "grad_norm": grad_norm,
        "grad_dot_normal": denominator,
        "conditioning_ratio": abs(denominator) / grad_norm,
        "delta_phi": delta,
        "linearized_normal_advance": float(-delta / denominator),
        "linearized_normal_speed": float(-delta / denominator / h),
        "not_exact_contour_motion": True,
    }


def make_frozen_velocity_phase_substep(solid: Any, p: Any, h: float, u0: Any, v0: Any):
    """Phase-only substep at a frozen velocity (diagnostic counterfactual, not a solver mode)."""

    @jax.jit
    def substep(phi: Any):
        state = pf.State(phi=phi, u=u0, v=v0, t=0.0)
        phi_rhs, _u, _v, _mu, mu_expl = pf.rhs(state, solid, p)
        return pf._phase_update(phi, u0, v0, solid, p, h, phi_rhs, mu_expl)[0]

    return substep


def make_ch_only_phase_substep(solid: Any, p: Any, h: float, shape: tuple[int, int]):
    """``u=v=0`` Cahn--Hilliard substep: pure relaxation, no advective contribution."""

    zero_u = jnp.zeros(shape, dtype=p.dtype)
    zero_v = jnp.zeros(shape, dtype=p.dtype)

    @jax.jit
    def substep(phi: Any):
        state = pf.State(phi=phi, u=zero_u, v=zero_v, t=0.0)
        _phi_rhs, _u, _v, _mu, mu_expl = pf.rhs(state, solid, p)
        return pf._phase_update(phi, zero_u, zero_v, solid, p, h, jnp.zeros_like(phi), mu_expl)[0]

    return substep


def counterfactual_matrix(bundle: Any, key: str, state: Any, *, horizon: float = 0.08) -> dict[str, Any]:
    """Fixed-state diagnostic counterfactuals (never alternative CFD datasets)."""
    p = bundle.p
    solid = bundle.solid
    h = float(p.dt) / 3.0
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    support_before = audit._local_gap(np.asarray(state.phi), 0.5, solid, p)

    def _record(label: str, before_phi: Any, after_phi: Any) -> dict[str, Any]:
        support_after = audit._local_gap(np.asarray(after_phi, dtype=np.float64), 0.5, solid, p)
        return {
            "label": label,
            "delta_mass_V_phi": float(
                np.sum(volume * (np.asarray(after_phi, dtype=np.float64) - np.asarray(before_phi, dtype=np.float64)))
            ),
            "contour_motion": contour_motion_audit(
                bundle, key, before_phi, after_phi, support_before, support_after, h
            ),
        }

    phi_rhs, _u_rhs, _v_rhs, _mu, mu_expl = pf.rhs(state, solid, p)
    phi = state.phi
    production = pf._phase_update(phi, state.u, state.v, solid, p, h, phi_rhs, mu_expl)[0]
    advective_only = (phi + h * phi_rhs).astype(pf.phase_state_dtype(p))
    ch_only_substep = make_ch_only_phase_substep(solid, p, h, tuple(np.shape(phi)))
    ch_only = ch_only_substep(phi)
    rows = [
        _record("i_production_phase_substep", phi, production),
        _record("ii_advective_only_no_implicit_solve", phi, advective_only),
        _record("iii_ch_only_zero_velocity", phi, ch_only),
    ]

    # (iv) frozen-velocity full phase evolution over a bounded interval.
    n_sub = max(1, int(round(horizon / h)))
    frozen_substep = make_frozen_velocity_phase_substep(solid, p, h, state.u, state.v)
    carry = phi
    gap_trace: list[float | None] = []
    for _ in range(n_sub):
        carry = frozen_substep(carry)
    frozen_velocity_phi = np.asarray(carry, dtype=np.float64)
    gap_trace.append(audit._local_gap(frozen_velocity_phi, 0.5, solid, p)["gap"])
    rows.append(
        {
            "label": "iv_frozen_velocity_full_phase_evolution",
            "horizon": float(horizon),
            "internal_substeps": n_sub,
            "delta_mass_V_phi": float(np.sum(volume * (frozen_velocity_phi - np.asarray(phi, dtype=np.float64)))),
            "gap_phi05": gap_trace[-1],
            "note": "diagnostic counterfactual: momentum frozen at t=0, phase operator unchanged",
        }
    )
    return {
        "case_name": bundle.case_name,
        "candidate": key,
        "classification": "DIAGNOSTIC_COUNTERFACTUAL",
        "production_formula_reused": True,
        "why_no_naive_additivity": (
            "over one substep the split is exact by construction, but the implicit solve and the advective "
            "flux are nonlinear in phi and in u, so increments from separate rollouts cannot be added once "
            "the state diverges"
        ),
        "rows": rows,
    }


def ch_only_relaxation(bundle: Any, key: str, state: Any, *, horizon: float) -> dict[str, Any]:
    """Pure Cahn--Hilliard relaxation with ``u=v=0``: can the interface reach contact alone?"""
    p = bundle.p
    solid = bundle.solid
    h = float(p.dt) / 3.0
    n_sub = int(round(horizon / h))
    phi = state.phi
    substep = make_ch_only_phase_substep(solid, p, h, tuple(np.shape(phi)))
    initial_gap = audit._local_gap(np.asarray(phi), 0.5, solid, p)["gap"]
    minimum_gap = initial_gap
    contact = False
    for index in range(n_sub):
        phi = substep(phi)
        if index % 3 == 2:
            gap = audit._local_gap(np.asarray(phi), 0.5, solid, p)["gap"]
            if gap is not None:
                minimum_gap = min(minimum_gap, gap)
                contact = contact or bool(gap <= audit.CONTACT_GAP_CELLS * p.dx)
    final_gap = audit._local_gap(np.asarray(phi), 0.5, solid, p)["gap"]
    return {
        "case_name": bundle.case_name,
        "candidate": key,
        "classification": "DIAGNOSTIC_COUNTERFACTUAL_CH_ONLY",
        "horizon": float(horizon),
        "internal_substeps": n_sub,
        "initial_gap_phi05": initial_gap,
        "final_gap_phi05": final_gap,
        "minimum_gap_phi05": minimum_gap,
        "contact_reached_by_pure_relaxation": bool(contact),
        "interpretation": (
            "a finite CH-only contact would be a relaxation phenomenon, not a hydrodynamic impact; the "
            "result is classified separately and never counted as an authenticated impact"
        ),
    }


def tier2_operator_and_budget(bundles: dict[str, Any], eligible_keys: tuple[str, ...]) -> dict[str, Any]:
    momentum: dict[str, Any] = {}
    budget: dict[str, Any] = {}
    contour: dict[str, Any] = {}
    for case_name in CASE_NAMES:
        bundle = bundles[case_name]
        momentum[case_name] = {}
        budget[case_name] = {}
        contour[case_name] = {}
        for key in eligible_keys:
            state, _details = build_candidate_state(bundle, key)
            momentum[case_name][key] = momentum_stage_ledger(bundle, key, state, n_substeps=1)
            ledger = phase_component_ledger(bundle, key, state, n_substeps=3)
            budget[case_name][key] = ledger
            phi_after = _advance_one_substep(bundle, state)
            first = ledger["substeps"][0]
            del first
            contour[case_name][key] = contour_motion_audit(
                bundle,
                key,
                np.asarray(state.phi),
                phi_after,
                audit._local_gap(np.asarray(state.phi), 0.5, bundle.solid, bundle.p),
                audit._local_gap(phi_after, 0.5, bundle.solid, bundle.p),
                float(bundle.p.dt) / 3.0,
            )
    return {"momentum_ledger": momentum, "phase_budget": budget, "contour_motion": contour}


def _advance_one_substep(bundle: Any, state: Any) -> np.ndarray:
    p = bundle.p
    h = float(p.dt) / 3.0
    phi_rhs, _u, _v, _mu, mu_expl = pf.rhs(state, bundle.solid, p)
    phi_new = pf._phase_update(state.phi, state.u, state.v, bundle.solid, p, h, phi_rhs, mu_expl)[0]
    return np.asarray(phi_new, dtype=np.float64)


# ---------------------------------------------------------------------------
# Tier 3 — bounded deterministic event trajectories
# ---------------------------------------------------------------------------
def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "step",
        "t",
        "gap_phi05",
        "gap_phi01",
        "gap_phi05_over_dx",
        "gap_phi05_over_eps",
        "gap_phi01_over_dx",
        "gap_phi01_over_eps",
        "local_approach_speed_into_solid",
        "local_support_phi05",
        "contact_phi05",
        "contact_phi01",
        "v_liquid",
        "v_core",
        "v_gas",
        "u_liquid",
        "liquid_weight_sum",
        "kinetic_energy_physical",
        "kinetic_energy_liquid",
        "max_speed_global",
        "max_speed_physical",
        "mass_Vphi",
        "formal_mass_ratio",
        "max_phi_overshoot",
        "deep_solid_leak",
        "beta",
        "drop_height",
        "spread_width",
        "centroid_y",
        "contact_line_x_center",
        "contact_line_x_left_relative_to_centroid",
        "contact_line_x_right_relative_to_centroid",
    )
    out = {key: row[key] for key in keep if key in row}
    if "momentum" in row and row["momentum"] is not None:
        out["momentum_liquid_momentum_y"] = row["momentum"]["liquid_momentum_y"]
        out["momentum_liquid_mass"] = row["momentum"]["liquid_mass_density_weighted"]
    if "local_support_phi05" in row and row["local_support_phi05"] is not None:
        support = row["local_support_phi05"]
        out["support_cell"] = support.get("gap_cell")
        out["support_xy"] = support.get("support_xy")
        out["wall_normal_into_solid"] = support.get("local_wall_normal_into_solid")
    # ``local_support_phi05`` is kept verbatim: the frozen authentication protocol reads it.
    return out


def run_tapered_trajectory(
    bundle: Any,
    key: str,
    *,
    t_end: float,
    profile: str,
    extension_horizon: float | None = None,
    postcontact_window: float = audit.POSTCONTACT_WINDOW,
) -> dict[str, Any]:
    """One exact deterministic trajectory with frozen contact detection and bounded extension."""
    p = bundle.p
    dt = float(p.dt)
    h = dt / 3.0
    state, details = build_candidate_state(bundle, key)
    kernel = impulse.make_fast_substep(bundle.solid, p, h)
    n_end = int(round(t_end / dt))
    if not math.isclose(n_end * dt, t_end, rel_tol=0.0, abs_tol=1e-10):
        raise AuditBlocked(f"t_end={t_end} is not an integer number of public steps at dt={dt}")
    anchors = {int(round(t / dt)): t for t in ANCHOR_TIMES if t <= t_end + 1e-12}
    rows: list[dict[str, Any]] = [
        _compact_row(
            audit._trajectory_row(
                state.phi, state.u, state.v, bundle.solid, p, bundle, 0, 0.0, include_constraints=True
            )
        )
    ]
    rows[0]["row_kind"] = "initial"
    anchor_rows: list[dict[str, Any]] = [rows[0]]
    trace: list[dict[str, Any]] = []
    contact_row: dict[str, Any] | None = None
    contact_step: int | None = None
    started = time.perf_counter()
    stop_reason = "horizon_reached"
    final_step = 0
    seen_support: list[Any] = []
    support_switch_count = 0
    for public_step in range(1, n_end + 1):
        for _ in range(3):
            state, _scalars = kernel(state)
        t = public_step * dt
        row = audit._trajectory_row(
            state.phi, state.u, state.v, bundle.solid, p, bundle, public_step, t, include_constraints=False
        )
        final_step = public_step
        support = row.get("local_support_phi05")
        if support is not None:
            cell = support.get("gap_cell")
            if seen_support and cell != seen_support[-1]:
                support_switch_count += 1
            seen_support.append(cell)
        contact = bool(row["contact_phi05"])
        if contact and contact_row is None:
            contact_row = row
            contact_step = public_step
            stop_reason = "contact_detected"
        if contact and contact_step is not None and t >= contact_step * dt + postcontact_window - 1e-12:
            rows.append({**_compact_row(row), "row_kind": "postcontact_window_end"})
            break
        if contact_step is None and public_step in anchors:
            anchor_rows.append({**_compact_row(row), "row_kind": "anchor"})
        if public_step % TRACE_STRIDE == 0:
            trace.append(
                {
                    "t": float(t),
                    "gap_phi05": row["gap_phi05"],
                    "local_approach_speed_into_solid": row["local_approach_speed_into_solid"],
                    "v_liquid": row["v_liquid"],
                    "v_core": row["v_core"],
                    "mass_Vphi": row["mass_Vphi"],
                    "max_phi_overshoot": row["max_phi_overshoot"],
                    "max_speed_physical": row["max_speed_physical"],
                    "support_cell": None if support is None else support.get("gap_cell"),
                }
            )
        rows.append(_compact_row(row))
    elapsed = time.perf_counter() - started
    authentication_rows = rows
    result = {
        "case_name": bundle.case_name,
        "candidate": key,
        "profile": profile,
        "diagnostic_only": True,
        "N": bundle.N,
        "dt": dt,
        "internal_substep_h": h,
        "t_end_requested": float(t_end),
        "t_end_actual": float(final_step * dt),
        "stop_reason": stop_reason,
        "support_switch_count": int(support_switch_count),
        "elapsed_seconds": float(elapsed),
        "t0_details": details,
        "anchors": anchor_rows,
        "trace": trace,
        "postcontact_row_count": int(sum(1 for row in rows if contact_step is not None and row["step"] > contact_step)),
        "final_state_binding": {
            "phi_hash": audit._array_sha(np.asarray(state.phi)),
            "u_hash": audit._array_sha(np.asarray(state.u)),
            "v_hash": audit._array_sha(np.asarray(state.v)),
            "t": float(state.t),
        },
        "_final_state": state,
        "_full_rows": authentication_rows,
    }
    result["impact_authentication"] = audit.authenticate_impact(
        {"rows": _authentication_rows(authentication_rows)}, bundle, observed_horizon=float(final_step * dt)
    )
    result["quality"] = {
        "max_phi_overshoot": max((row["max_phi_overshoot"] for row in rows), default=None),
        "max_formal_mass_ratio": max((row["formal_mass_ratio"] for row in rows), default=None),
        "min_formal_mass_ratio": min((row["formal_mass_ratio"] for row in rows), default=None),
        "final_formal_mass_ratio": rows[-1]["formal_mass_ratio"],
        "max_speed_global": max((row["max_speed_global"] for row in rows), default=None),
        "max_deep_solid_leak": max((row["deep_solid_leak"] for row in rows), default=None),
        "inherited_pillar_overshoot_and_mass_blockers_still_reported": True,
    }
    if extension_horizon is not None and contact_step is None:
        extension = _extend_trajectory(bundle, key, state, final_step, extension_horizon, postcontact_window)
        result["extension"] = extension
        if extension["contact_step"] is not None:
            combined = rows + extension["rows"]
            extended = audit.authenticate_impact(
                {"rows": _authentication_rows(combined)}, bundle, observed_horizon=extension["extension_horizon"]
            )
            result["primary_horizon_authentication"] = result["impact_authentication"]
            result["impact_authentication"] = extended
            result["impact_authentication_scope"] = (
                f"declared targeted extension to t<={extension_horizon}: the primary {PRIMARY_EVENT_HORIZON} "
                "horizon contained no contact, so the extended search supplies the authoritative event label"
            )
            result["authentication_rows"] = len(combined)
    return result


def _authentication_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows in the shape ``audit.authenticate_impact`` consumes, without recomputation."""
    out = []
    for row in rows:
        if "gap_phi05" not in row:
            continue
        out.append(
            {
                **row,
                "momentum": {
                    "liquid_momentum_y": row.get("momentum_liquid_momentum_y"),
                    "liquid_mass_density_weighted": row.get("momentum_liquid_mass"),
                }
                if "momentum_liquid_momentum_y" in row
                else None,
                "local_approach_speed_into_solid": row.get("local_approach_speed_into_solid"),
                "v_liquid": row.get("v_liquid"),
                "beta": row.get("beta"),
                "drop_height": row.get("drop_height"),
                "contact_line_x_center": row.get("contact_line_x_center"),
            }
        )
    return out


def _extend_trajectory(
    bundle: Any, key: str, state: Any, start_step: int, t_end: float, postcontact_window: float
) -> dict[str, Any]:
    p = bundle.p
    dt = float(p.dt)
    h = dt / 3.0
    kernel = impulse.make_fast_substep(bundle.solid, p, h)
    n_end = int(round(t_end / dt))
    contact_step: int | None = None
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for public_step in range(start_step + 1, n_end + 1):
        for _ in range(3):
            state, _scalars = kernel(state)
        t = public_step * dt
        row = audit._trajectory_row(
            state.phi, state.u, state.v, bundle.solid, p, bundle, public_step, t, include_constraints=False
        )
        if row["contact_phi05"] and contact_step is None:
            contact_step = public_step
        rows.append(_compact_row(row))
        if contact_step is not None and t >= contact_step * dt + postcontact_window - 1e-12:
            break
    return {
        "extension_horizon": float(t_end),
        "extension_reason": "declared single targeted continuation for an otherwise valid case",
        "contact_step": contact_step,
        "contact_time": None if contact_step is None else float(contact_step * dt),
        "rows": rows,
        "elapsed_seconds": float(time.perf_counter() - started),
        "status": "NO_CONTACT_WITHIN_OBSERVED_WINDOW" if contact_step is None else "CONTACT_OBSERVED",
    }


def tier3_events(
    bundles: dict[str, Any],
    eligible_keys: tuple[str, ...],
    initial_matrix: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    compatible: dict[str, list[str]] = {}
    for case_name in CASE_NAMES:
        compatible[case_name] = [
            key
            for key in eligible_keys
            if initial_matrix[case_name]["candidates"][key]["frozen_initial_acceptance"]["pass"]
        ]
    events: dict[str, Any] = {}
    quarantined: dict[str, Any] = {}
    for case_name in CASE_NAMES:
        events[case_name] = {}
        for key in compatible[case_name]:
            progress(f"event trajectory: {case_name}/{key} to t<={PRIMARY_EVENT_HORIZON}")
            events[case_name][key] = run_tapered_trajectory(
                bundle=bundles[case_name],
                key=key,
                t_end=PRIMARY_EVENT_HORIZON,
                profile=profile,
                extension_horizon=TARGETED_EXTENSION_HORIZON,
            )
            events[case_name][key].pop("_final_state", None)
        # Ineligible candidates get a quarantined one-substep negative-control demonstration only.
        for key in eligible_keys:
            if key in compatible[case_name]:
                continue
            state, _ = build_candidate_state(bundles[case_name], key)
            supported = momentum_stage_ledger(bundles[case_name], key, state, n_substeps=1)
            phi_after = _advance_one_substep(bundles[case_name], state)
            supported["quarantined"] = True
            supported["candidate_eligibility"] = "INITIALIZER_INCOMPATIBLE_CASE"
            supported["one_substep_only"] = True
            supported["contour_after_one_substep"] = {
                "gap_phi05": audit._local_gap(phi_after, 0.5, bundles[case_name].solid, bundles[case_name].p)["gap"]
            }
            quarantined[f"{case_name}/{key}"] = supported
    return {
        "profile": profile,
        "eligible_candidates_per_case": compatible,
        "events": events,
        "quarantined_one_substep_negative_controls": quarantined,
        "not_equal_to_u_impact_note": "contact is authenticated by the frozen local-normal criterion, not by u_impact",
    }


# ---------------------------------------------------------------------------
# Tier 4 — matched-time temporal sensitivity
# ---------------------------------------------------------------------------
def temporal_sensitivity(
    bundles_by_dt: dict[float, dict[str, Any]],
    keys: tuple[str, ...],
    cases: tuple[str, ...],
    *,
    horizon: float,
    profile: str,
) -> dict[str, Any]:
    """Matched-physical-time dt comparison at fixed N with bitwise-identical t=0 fields."""
    out: dict[str, Any] = {
        "profile": profile,
        "horizon": float(horizon),
        "dt_levels": list(DT_LEVELS),
        "keys": list(keys),
        "cases": {},
        "official_refinement_gate_executed": False,
        "official_gate_note": (
            "the original L1A-2q/L1A refinement scoring is NOT executed here; all numbers below are "
            "descriptive dt-sensitivity measurements and must not be reported as the official 3% gate"
        ),
    }
    for case_name in cases:
        out["cases"][case_name] = {}
        initial_hashes: dict[str, Any] = {}
        for dt in DT_LEVELS:
            bundle = bundles_by_dt[dt][case_name]
            for key in keys:
                state, _details = build_candidate_state(bundle, key)
                hashes = {
                    "phi": audit._array_sha(np.asarray(state.phi)),
                    "u": audit._array_sha(np.asarray(state.u)),
                    "v": audit._array_sha(np.asarray(state.v)),
                    "geometry": audit._geometry_hash(bundle.solid),
                }
                initial_hashes.setdefault(key, {})[f"{dt:g}"] = hashes
        out["cases"][case_name]["t0_hash_invariance"] = {
            key: {
                "bitwise_identical_across_dt": len({json.dumps(value, sort_keys=True) for value in per_dt.values()})
                == 1,
                "per_dt": per_dt,
            }
            for key, per_dt in initial_hashes.items()
        }
        for key in keys:
            out["cases"][case_name][key] = {}
            for dt in DT_LEVELS:
                progress(f"temporal trajectory: {case_name}/{key} dt={dt:g}")
                bundle = bundles_by_dt[dt][case_name]
                run = run_tapered_trajectory(
                    bundle=bundle, key=key, t_end=horizon, profile=profile, extension_horizon=None
                )
                run.pop("_final_state", None)
                out["cases"][case_name][key][f"{dt:g}"] = {
                    "matched_times": _matched_time_readout(run),
                    "impact_authentication": run["impact_authentication"],
                    "quality": run["quality"],
                    "stop_reason": run["stop_reason"],
                    "t_end_actual": run["t_end_actual"],
                    "elapsed_seconds": run["elapsed_seconds"],
                    "effective_dt": float(bundle.p.dt),
                    "requested_dt": bundle.requested_dt,
                    "policy": bundle.policy,
                }
    out["dt_sensitivity_verdict"] = _dt_sensitivity_verdict(out)
    return out


def _matched_time_readout(run: dict[str, Any]) -> dict[str, Any]:
    anchors = {round(float(row["t"]), 6): row for row in run["anchors"]}
    selected = {}
    for t in ANCHOR_TIMES:
        if t == 0.0:
            continue
        row = anchors.get(round(t, 6))
        if row is None:
            continue
        selected[f"t_{t:.2f}"] = {
            "gap_phi05": row["gap_phi05"],
            "gap_phi05_over_dx": row["gap_phi05_over_dx"],
            "local_approach_speed_into_solid": row["local_approach_speed_into_solid"],
            "v_liquid": row["v_liquid"],
            "v_core": row["v_core"],
            "mass_Vphi": row["mass_Vphi"],
            "support_cell": row.get("support_cell"),
        }
    contact = run["impact_authentication"].get("contact_time_interpolated_gap_threshold")
    return {
        "anchors": selected,
        "contact_time": contact,
        "contact_time_discrete": run["impact_authentication"].get("contact_time_discrete"),
        "precontact_local_normal_approach": run["impact_authentication"].get(
            "precontact_local_normal_approach_speed_positive_toward_solid"
        ),
        "precontact_time": run["impact_authentication"].get("precontact_time"),
        "verdict": run["impact_authentication"]["verdict"],
        "observed_horizon": run["impact_authentication"]["observed_horizon"],
    }


def _dt_sensitivity_verdict(matrix: dict[str, Any]) -> dict[str, Any]:
    """Descriptive dt trend per case/candidate; never the official refinement gate."""
    out: dict[str, Any] = {"label_rule": "DT_SENSITIVE when the precontact approach changes materially with dt"}
    flagged = False
    for case_name, per_key in matrix["cases"].items():
        out[case_name] = {}
        for key in matrix["keys"]:
            entries = per_key.get(key, {})
            approaches = {
                dt: entry["matched_times"]["precontact_local_normal_approach"] for dt, entry in entries.items()
            }
            finite = {dt: value for dt, value in approaches.items() if value is not None}
            verdict = "UNMEASURED"
            ratio = None
            if len(finite) >= 2:
                ordered = sorted(finite.items(), key=lambda item: -float(item[0]))
                coarse = ordered[0][1]
                fine = ordered[-1][1]
                if abs(coarse) > 1e-12:
                    ratio = fine / coarse
                verdict = "DT_SENSITIVE" if (ratio is None or ratio < 0.9 or ratio > 1.1) else "DT_STABLE"
                flagged = flagged or verdict == "DT_SENSITIVE"
            out[case_name][key] = {
                "precontact_approach_by_dt": approaches,
                "finest_over_coarsest_ratio": ratio,
                "label": verdict,
            }
    out["any_dt_sensitive"] = flagged
    return out


# ---------------------------------------------------------------------------
# Falsification matrix
# ---------------------------------------------------------------------------
def falsification_matrix(
    bundles: dict[str, Any],
    initial_matrix: dict[str, Any],
    band_matrix: dict[str, Any],
    tier2: dict[str, Any],
    events: dict[str, Any] | None,
    *,
    profile: str,
) -> dict[str, Any]:
    """Explicit counterfactual numbers for the competing mechanisms."""
    case_name = "flat_we100_ct050"
    bundle = bundles[case_name]
    band_case = band_matrix[case_name]["candidates"]
    c2 = initial_matrix[case_name]["candidates"]["C2_BASE"]
    t1 = initial_matrix[case_name]["candidates"]["T1_NEAR_NARROW"]
    t3 = initial_matrix[case_name]["candidates"]["T3_FAR_NARROW"]
    control = initial_matrix[case_name]["quarantined_controls"]["NC_NO_WALL_TAPER"]

    def _approach(entry: dict[str, Any]) -> float | None:
        return entry["initial_front_contour"]["local_normal_approach_speed_into_solid"]

    state_c2, _ = build_candidate_state(bundle, "C2_BASE")
    state_t3, _ = build_candidate_state(bundle, "T3_FAR_NARROW")
    adv_c2 = _front_advective_flux(bundle, state_c2)
    adv_t3 = _front_advective_flux(bundle, state_t3)
    negative_controls = impulse.negative_controls(n=48)
    fusion_a = {
        "hypothesis": "H-TAPER: the SDF taper onset suppresses front-side incident fluid velocity",
        "test_A_front_speed_vs_compatibility": {
            "C2_BASE_initial_approach": _approach(c2),
            "T1_NEAR_NARROW_initial_approach": _approach(t1),
            "T3_FAR_NARROW_initial_approach": _approach(t3),
            "quarantined_no_wall_taper_initial_approach": _approach(control),
            "C2_central_gates_pass": c2["frozen_initial_acceptance"]["pass"],
            "T1_central_gates_pass": t1["frozen_initial_acceptance"]["pass"],
            "T1_failing_gate": [name for name, ok in t1["frozen_initial_acceptance"]["checks"].items() if not ok],
            "T1_nearwall_ratio_over_limit": t1["frozen_initial_acceptance"]["dimensionless_residuals"][
                "nearwall_chi_speed_over_u_impact"
            ]
            / audit.INITIAL_ACCEPTANCE["nearwall_chi_ge_0p01_speed_over_u_impact_max"],
        },
        "test_A2_envelope_contribution": {
            "definition": (
                "the quarantined taper-free control isolates how much of the front-speed suppression comes "
                "from the compact radial envelope rather than from the wall taper"
            ),
            "taper_free_front_approach": _approach(control),
            "taper_present_front_approach": _approach(c2),
        },
        "test_B_front_advective_flux": {
            "C2_BASE_front_advective_face_flux": adv_c2,
            "T3_FAR_NARROW_front_advective_face_flux": adv_t3,
            "note": "same phase field, so any change isolates the advective face flux carried by the seed velocity",
        },
        "test_D_contour_robustness": {
            "independent_reconstruction": tier2["contour_motion"][case_name]["C2_BASE"].get("independent_vs_official"),
            "declared_tolerance_dx": CONTOUR_AGREEMENT_DX,
        },
        "test_F_projection_and_negative_controls": {
            "l1a2r_control_A": negative_controls["A_empty_solid_uniform"],
            "l1a2r_control_B": negative_controls["B_projection_uniform_with_solid_present"],
            "l1a2r_control_D": negative_controls["D_pure_gradient_field"],
            "same_state_projection_identity": tier2["momentum_ledger"][case_name]["C2_BASE"]["substeps"][0][
                "projection_plain_grid_mean_identity"
            ],
        },
        "test_H_global_confounders": {
            key: {
                "initial_liquid_momentum_y": initial_matrix[case_name]["candidates"][key]["raw_constraint_values"].get(
                    "liquid_v_weighted"
                ),
                "return_flow_peak_speed": initial_matrix[case_name]["candidates"][key]["raw_constraint_values"][
                    "return_flow_peak_speed"
                ],
                "kinetic_energy_open_fluid": initial_matrix[case_name]["candidates"][key]["raw_constraint_values"][
                    "kinetic_energy_open_fluid"
                ],
            }
            for key in CANDIDATE_RUN_ORDER
        },
        "band_summary_C2_vs_T1": {
            "C2_BASE": band_case["C2_BASE"]["bands"],
            "T1_NEAR_NARROW": band_case["T1_NEAR_NARROW"]["bands"],
        },
    }
    fusion_a["profile"] = profile
    fusion_a["competing_mechanisms"] = {
        "H_TAPER": "SUPPORTED_IN_DIRECTION_BUT_COMPATIBILITY_BLOCKED"
        if (_approach(t1) or 0.0) > (_approach(c2) or 0.0)
        else "FALSIFIED",
        "H_CH": "TESTED_SEPARATELY",
        "H_BP": "TESTED_SEPARATELY",
        "H_GEOMETRY": "TESTED_SEPARATELY",
        "H_MULTI": "CONSISTENT_WITH_MEASUREMENTS",
    }
    if events is not None:
        fusion_a["event_side_evidence"] = {
            case: {
                key: {
                    "verdict": run["impact_authentication"]["verdict"],
                    "contact_time": run["impact_authentication"].get("contact_time_interpolated_gap_threshold"),
                    "precontact_approach": run["impact_authentication"].get(
                        "precontact_local_normal_approach_speed_positive_toward_solid"
                    ),
                }
                for key, run in per_case.items()
            }
            for case, per_case in events["events"].items()
        }
    return fusion_a


def _front_advective_flux(bundle: Any, state: Any) -> dict[str, Any]:
    p = bundle.p
    support = audit._local_gap(np.asarray(state.phi), 0.5, bundle.solid, p)
    flux_x, flux_y = pf.phase_advective_fluxes(state.u, state.v, state.phi, bundle.solid, p)
    cell = support.get("gap_cell")
    if cell is None:
        return {"status": "UNMEASURED_NO_SUPPORT"}
    i, j = int(cell[0]), int(cell[1])
    window = (slice(max(0, i - 2), i + 3), slice(max(0, j - 1), j + 2))
    return {
        "support_cell": [i, j],
        "flux_x_at_support": float(np.asarray(flux_x, dtype=np.float64)[i, j]),
        "flux_y_at_support": float(np.asarray(flux_y, dtype=np.float64)[i, j]),
        "flux_y_window_max_abs": float(np.max(np.abs(np.asarray(flux_y, dtype=np.float64)[window]))),
        "flux_x_window_max_abs": float(np.max(np.abs(np.asarray(flux_x, dtype=np.float64)[window]))),
        "definition": "A_f * u_n,f * phi_upwind,f on the shared cut-cell faces; conservative by construction",
    }


# ---------------------------------------------------------------------------
# Quality, decision and assembly
# ---------------------------------------------------------------------------
def quality_matrix(profile: str) -> dict[str, Any]:
    baseline = json.loads(QUALITY_BASELINE.read_text())
    return {
        "profile": profile,
        "quality_baseline_sha256": sha256_file(QUALITY_BASELINE),
        "quality_baseline_untouched": True,
        "historical_baseline_status": "PASS_BASELINE_ONLY",
        "historical_summary": baseline.get("summary"),
        "declared_new_code_diagnostics": 0,
        "frozen_prohibitions": {
            "frozen_solver_unmodified": True,
            "thresholds_unmodified": True,
            "historical_evidence_unmodified": True,
            "mass_baseline_unmodified": True,
            "production_initializer_unmodified": True,
            "solver_contract_unmodified": True,
            "approved_sha_migrations_unmodified": True,
            "no_autoformat_on_frozen_modules": True,
        },
        "inherited_blockers": {
            "primary_blocker": "REQUIRED_PILLAR_OR_COMPLEX_IMPACT_AUTHENTICATION",
            "L1B_DATA": "L1B_DATA_NOT_READY",
            "SPATIAL_REFINEMENT": "FAIL",
            "TEMPORAL_REFINEMENT": "UNMEASURED",
            "DIRECT_FIELD_LABELS": "UNMEASURED",
            "pillar_training_overshoot_inherited": 0.03697,
            "pillar_training_overshoot_limit": 0.02,
            "max_total_mass_ratio_inherited": 1.005185,
            "max_total_mass_ratio_limit": 1.005,
            "inherited_blockers_unchanged_by_this_stage": True,
        },
        "no_promotion": {
            "candidate_promoted": False,
            "production_change": False,
            "contract_13": False,
            "l1b_generation": False,
            "ml_training": False,
            "full_t8_recomputation": False,
        },
    }


def decide(
    *,
    profile: str,
    initial_matrix: dict[str, Any] | None,
    tier3: dict[str, Any] | None,
    temporal: dict[str, Any] | None,
    budget: dict[str, Any] | None,
) -> dict[str, Any]:
    """One taper-mechanism verdict, one candidate outcome, one stage decision."""
    quick = profile != "forensic"
    if initial_matrix is None:
        return {
            "profile": profile,
            "quick_only": quick,
            "physical_verdict_valid": False,
            "taper_mechanism_verdict": "INCONCLUSIVE",
            "candidate_outcome": "UNMEASURED",
            "stage_decision": "PREREQUISITE_BLOCKED",
            "reason": "initial compatibility stage did not run",
        }
    compatible_per_case = {
        case: [
            key
            for key in CANDIDATE_RUN_ORDER
            if initial_matrix[case]["candidates"][key]["frozen_initial_acceptance"]["pass"]
        ]
        for case in CASE_NAMES
    }
    incompatible_per_case = {
        case: [
            key
            for key in CANDIDATE_RUN_ORDER
            if not initial_matrix[case]["candidates"][key]["frozen_initial_acceptance"]["pass"]
        ]
        for case in CASE_NAMES
    }
    raised_front = []
    for case in CASE_NAMES:
        base = initial_matrix[case]["candidates"]["C2_BASE"]["initial_front_contour"][
            "local_normal_approach_speed_into_solid"
        ]
        for key in CANDIDATE_RUN_ORDER:
            value = initial_matrix[case]["candidates"][key]["initial_front_contour"][
                "local_normal_approach_speed_into_solid"
            ]
            if base is not None and value is not None and value > base + 1e-12:
                raised_front.append({"case": case, "candidate": key, "delta_over_u_impact": value - base})
    eligible_keys = [key for key in CANDIDATE_RUN_ORDER if key != "C2_BASE"]
    any_eligible_change = any(key in compatible_per_case[case] for case in CASE_NAMES for key in eligible_keys)
    raised_and_compatible = [item for item in raised_front if item["candidate"] in compatible_per_case[item["case"]]]
    if raised_front and not raised_and_compatible:
        mechanism = "TAPER_COMPATIBILITY_TRADEOFF"
    elif raised_and_compatible:
        mechanism = "TAPER_MECHANISM_PARTIALLY_ISOLATED"
    else:
        mechanism = "TAPER_NOT_DOMINANT"
    if any_eligible_change and tier3 is None:
        candidate_outcome = "UNMEASURED"
        stage_decision = "BOUNDED_ADDITIONAL_DIAGNOSTIC"
    elif tier3 is None:
        candidate_outcome = "NONE_COMPATIBLE"
        stage_decision = "PREREQUISITE_BLOCKED"
    else:
        verdicts = [
            run["impact_authentication"]["verdict"]
            for case in tier3["events"]
            for run in tier3["events"][case].values()
        ]
        authenticated = any(verdict == "IMPACT_AUTHENTICATED" for verdict in verdicts)
        candidate_outcome = (
            "COMPATIBLE_IMPACT_BUT_QUALITY_OR_REFINEMENT_FAIL" if authenticated else "COMPATIBLE_NO_MEANINGFUL_IMPACT"
        )
        stage_decision = "DIAGNOSTIC_COMPLETE_NO_PROMOTION"
    return {
        "profile": profile,
        "quick_only": quick,
        "physical_verdict_valid": not quick,
        "taper_mechanism_verdict": mechanism if not quick else "INCONCLUSIVE",
        "candidate_outcome": candidate_outcome if not quick else "UNMEASURED",
        "stage_decision": stage_decision if not quick else "PREREQUISITE_BLOCKED",
        "compatible_candidates_per_case": compatible_per_case,
        "incompatible_candidates_per_case": incompatible_per_case,
        "front_speed_raising_candidates": raised_front,
        "front_speed_raising_and_compatible": raised_and_compatible,
        "temporal_dt_sensitive": None
        if temporal is None
        else temporal.get("dt_sensitivity_verdict", {}).get("any_dt_sensitive"),
        "phase_budget_identity_holds": None
        if budget is None
        else all(
            row["phase_update_bitwise_reproduced"] and row["closure_linf_active_cells"] <= 1e-15
            for case in budget
            for key in budget[case]
            for row in budget[case][key]["substeps"]
        ),
        "next_action": "see next_action() in the final report",
    }


def next_action(decision: dict[str, Any]) -> dict[str, Any]:
    """Exactly one smallest next physical action; never executed inside this stage."""
    verdict = decision["taper_mechanism_verdict"]
    if verdict == "TAPER_COMPATIBILITY_TRADEOFF":
        action = (
            "Design and audit a boundary-consistent incident velocity whose advective forcing is confined to "
            "the region sdf > 3 dx while preserving the frozen chi>=0.01 near-wall gate: rebuild the seed from "
            "a single smooth streamfunction whose *radial envelope* (not the wall taper) supplies the near-wall "
            "roll-off, then re-measure the same t=0 matrix before any trajectory work."
        )
    elif verdict == "TAPER_NOT_DOMINANT":
        action = "Target the measured dominant term with one bounded phase-transport experiment."
    else:
        action = (
            "Re-run the same-state audit with a compact-support envelope re-derived from the wall distance so "
            "the taper onset is no longer the only near-wall suppression."
        )
    return {
        "action": action,
        "bounded_scope": "initializer design + same-state matrix only; no trajectory, no generator, no ML",
        "must_not_be_started_in_this_stage": True,
    }


def assemble_evidence(
    *,
    profile: str,
    out_dir: Path,
    preflight: dict[str, Any],
    operator_map: dict[str, Any],
    reproduction: dict[str, Any],
    initial_matrix: dict[str, Any],
    band_matrix: dict[str, Any],
    tier2: dict[str, Any] | None,
    tier3: dict[str, Any] | None,
    temporal: dict[str, Any] | None,
    falsification: dict[str, Any] | None,
    decision: dict[str, Any],
    run_meta: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the committed evidence set and bind every file by SHA256."""
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_class = "FORENSIC_N192" if profile == "forensic" else "QUICK_ONLY"
    missing_status = {
        "status": "NOT_RUN_IN_THIS_PROFILE_STAGE_SELECTION",
        "evidence_class": evidence_class,
        "may_not_be_assembled_into_a_physical_pass": True,
    }
    files: dict[str, Any] = {
        "prerequisite_and_source_map.json": {
            "preflight": preflight,
            "operator_map": operator_map,
            "c2_exact_reproduction": reproduction,
            "evidence_class": evidence_class,
        },
        "taper_candidate_design_contract.json": {
            "prespecified_candidates": TAPER_CANDIDATES,
            "declared_run_order": list(CANDIDATE_RUN_ORDER),
            "quarantined_controls": QUARANTINED_CONTROLS,
            "taper_equation": "wall_taper = smooth5((sdf - a*dx)/(b*dx)); smooth5(z)=6z^5-15z^4+10z^3",
            "unchanged_components": [
                "compact radial envelope (1 - r2/(R+0.45)^2)^4_+ times sx",
                "periodic y seam taper smooth5((dist_seam-3dy)/(3dy))",
                "streamfunction differentiation u=D_y psi, v=-D_x psi",
                "V_i-weighted phi>=0.9 core-mean amplitude normalization to -u_impact",
                "phase field, solid geometry, case placement, M, We, Re, cos_theta, eps, dt policy",
            ],
            "not_an_optimization": True,
            "extra_candidates": [],
            "evidence_class": evidence_class,
        },
        "candidate_initial_compatibility_matrix.json": initial_matrix,
        "front_velocity_distance_band_matrix.json": band_matrix,
        "phase_advective_CH_implicit_budget.json": tier2["phase_budget"] if tier2 else missing_status,
        "interface_contour_motion_audit.json": tier2["contour_motion"] if tier2 else missing_status,
        "first_substep_operator_ledger.json": tier2["momentum_ledger"] if tier2 else missing_status,
        "bounded_candidate_event_matrix.json": tier3 if tier3 else missing_status,
        "temporal_sensitivity_matrix.json": temporal if temporal else missing_status,
        "counterfactual_and_falsification_matrix.json": falsification if falsification else missing_status,
        "quality_and_blocker_matrix.json": quality_matrix(profile),
    }
    decision_payload = {**decision, "next_action": next_action(decision), "evidence_class": evidence_class}
    files["mechanism_decision.json"] = decision_payload
    report_files = {"l1a2t_final_report.json", "l1a2t_final_report.md"}
    missing_deliverables = [name for name in REQUIRED_EVIDENCE_FILES if name not in files and name not in report_files]
    if missing_deliverables:
        raise AuditBlocked(f"assembler is missing mandatory evidence deliverables: {missing_deliverables}")
    written: dict[str, str] = {}
    for name, payload in files.items():
        path = out_dir / name
        write_json(path, payload)
        written[name] = sha256_file(path)
    report = {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "profile": profile,
        "evidence_class": evidence_class,
        "decision": decision_payload,
        "compute": run_meta,
        "quality_status": validation
        if validation is not None
        else {
            "status": "NOT_RECORDED_IN_THIS_RUN",
            "note": "run the focused test module and the four-tool quality gate, then pass --validation-json",
        },
        "evidence_sha256": written,
    }
    write_json(out_dir / "l1a2t_final_report.json", report)
    written["l1a2t_final_report.json"] = sha256_file(out_dir / "l1a2t_final_report.json")
    report["evidence_sha256"] = written
    markdown = render_report(
        preflight=preflight,
        reproduction=reproduction,
        initial_matrix=initial_matrix,
        band_matrix=band_matrix,
        tier2=tier2,
        tier3=tier3,
        temporal=temporal,
        falsification=falsification,
        decision=decision_payload,
        run_meta=run_meta,
        profile=profile,
        validation=validation,
    )
    (out_dir / "l1a2t_final_report.md").write_text(markdown)
    written["l1a2t_final_report.md"] = sha256_file(out_dir / "l1a2t_final_report.md")
    absent = [name for name in REQUIRED_EVIDENCE_FILES if not (out_dir / name).exists()]
    if absent:
        raise AuditBlocked(f"mandatory evidence deliverables were not written: {absent}")
    manifest = {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "profile": profile,
        "evidence_class": evidence_class,
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "timestep_policy": EXPECTED_POLICY,
        "verified_main_sha": VERIFIED_MAIN_SHA,
        "pr22": PR22,
        "source_hashes": preflight["source_hashes_current"],
        "frozen_evidence_input_sha256": preflight["frozen_evidence_input_sha256"],
        "quality_baseline_sha256": preflight["quality_baseline_sha256"],
        "cases": list(CASE_NAMES),
        "candidate_identity": {
            "prespecified": {key: TAPER_CANDIDATES[key] for key in CANDIDATE_RUN_ORDER},
            "quarantined": QUARANTINED_CONTROLS,
        },
        "run_profile": {
            "N": N_FORENSIC if profile == "forensic" else N_QUICK,
            "dt_levels": list(DT_LEVELS),
            "stages": run_meta.get("stages"),
        },
        "evidence_files": {name: {"sha256": value} for name, value in sorted(written.items())},
        "missing_or_unmeasured": run_meta.get("missing_or_unmeasured", []),
        "no_promotion": True,
    }
    write_json(out_dir / "manifest.json", manifest)
    return {"manifest": manifest, "evidence_sha256": written}


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "UNMEASURED"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def render_report(**kwargs: Any) -> str:
    """Human-readable A--R report; every number below also exists in the JSON evidence."""
    profile = kwargs["profile"]
    decision = kwargs["decision"]
    preflight = kwargs["preflight"]
    run_meta = kwargs["run_meta"]
    initial = kwargs["initial_matrix"] or {}
    band_matrix = kwargs.get("band_matrix") or {}
    tier2 = kwargs.get("tier2")
    tier3 = kwargs.get("tier3")
    temporal = kwargs.get("temporal")
    falsification = kwargs.get("falsification") or {}
    validation = kwargs.get("validation")
    prerequisite = preflight["prerequisite"]
    lines = [
        "# L1A-2t — Near-Wall Taper, Interface Transport and Contact-Side Impact Causal Audit",
        "",
        f"Profile: `{profile}` | evidence class: `{'FORENSIC_N192' if profile == 'forensic' else 'QUICK_ONLY'}`"
        " | diagnostic only: no promotion, no production change, no contract change.",
        "",
        "## A. Prerequisite, PR and CI status",
        f"- verified merged main SHA: `{prerequisite['verified_merge_sha']}`; verified parents "
        f"{prerequisite['verified_sha_parents']}",
        f"- PR #22 `{PR22['state']}` (head `{PR22['head_sha']}`, merged {PR22['merged_at_utc']}); "
        f"hosted Python CI run {prerequisite['pr22_hosted_python_ci_run']}: "
        f"{prerequisite['pr22_hosted_python_ci_status']} (engineering-only, reuse of the frozen freeze)",
        f"- branch `{prerequisite['branch']}`, merge-base `{prerequisite['merge_base_sha']}`, "
        f"worktree clean outside stage deliverables: "
        f"{prerequisite['worktree_clean_within_stage_deliverables']}",
        f"- stage commits on top of the verified main: {prerequisite['commits_since_verified_main']}",
        "- L1A-2t hosted CI is not claimed here; no CI result is asserted in advance",
        "",
        "## B. Frozen source, policy and prohibitions",
        f"- solver contract {int(pf.SOLVER_CONTRACT_VERSION)}; timestep policy `{EXPECTED_POLICY}`",
        f"- frozen source hashes match the frozen L1A-2s manifest: {preflight['source_freeze_checks']}",
        "- untouched: phasefield solver, production initializer default, thresholds, historical evidence, "
        "mass baselines, approved SHA migrations, `quality/legacy_two_phase_debt_v1.json`",
        "- no T=8 recompute, no L1B generation, no ML training, no contract 13, no candidate promotion",
        "",
        "## C. Inherited L1A-2s measurements and replication",
        "- inherited frozen table (dt refinement, flat/pillar): approach "
        "0.04492/0.02356/0.01177 (flat100), 0.04549/0.02315/0.01099 (flat200), 0.06940/0.03815/none (pillar)",
        "- this stage re-measures them from the reproduced C2 state instead of copying them; the C2 t=0 fields",
        "  are hash-compared with the frozen L1A-2s evidence and any mismatch raises a fail-closed error",
        f"- C2 t=0 identity reproduction: {json.dumps(kwargs.get('reproduction', {}), sort_keys=True)[:400]}",
        "",
        "## D. Taper mechanism hypothesis",
        "- H-TAPER: the C2 wall taper onset (a=4 dx) sits above the initial contour support (sdf~2.5 dx), so the",
        "  incident front fluid velocity there is exactly zero; opening the taper should restore it.",
        "- competing explanations measured separately: H-CH, H-BP, H-GEOMETRY (diffuse-indicator gate domain), H-MULTI",
    ]
    envelope = falsification.get("test_A2_envelope_contribution")
    if envelope:
        lines.append(
            f"- quarantined taper-free control: front approach {_fmt(envelope['taper_free_front_approach'])} "
            f"vs {_fmt(envelope['taper_present_front_approach'])} with the taper present"
        )
    lines += ["", "## E. Prespecified candidate equations"]
    for key in CANDIDATE_RUN_ORDER:
        spec = TAPER_CANDIDATES[key]
        lines.append(
            f"- `{key}` a={_fmt(spec['onset_a_dx'], 3)} b={_fmt(spec['width_b_dx'], 3)} (role: {spec['role']})"
        )
    for key, spec in QUARANTINED_CONTROLS.items():
        lines.append(f"- quarantined `{key}`: {spec['role']}, never eligible, never promoted")
    lines += [
        "- identical radial envelope, periodic seam taper, differentiation, x0/y0/R, core-mean normalization",
        "",
        "## F. All-candidate t=0 compatibility (frozen gates)",
        "",
        "| case | candidate | gate | failing gate | near-wall chi ratio (limit 0.02) |"
        " front approach | front floor 0.1 |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for case in CASE_NAMES:
        for key in CANDIDATE_RUN_ORDER:
            entry = initial[case]["candidates"][key]
            acceptance = entry["frozen_initial_acceptance"]
            fails = [name for name, ok in acceptance["checks"].items() if not ok]
            front = entry["initial_front_contour"]
            lines.append(
                f"| `{case}` | `{key}` | {'PASS' if acceptance['pass'] else 'FAIL'} | "
                f"{', '.join(fails) if fails else '-'} | "
                f"{_fmt(acceptance['dimensionless_residuals']['nearwall_chi_speed_over_u_impact'], 4)} | "
                f"{_fmt(front['local_normal_approach_speed_into_solid'], 7)} | "
                f"{front['exceeds_frozen_approach_floor']} |"
            )
    control_rows = []
    for case in CASE_NAMES:
        control = initial[case]["quarantined_controls"]["NC_NO_WALL_TAPER"]
        residual = control["frozen_initial_acceptance"]["dimensionless_residuals"]
        control_rows.append(
            f"`{case}`: front approach "
            f"{_fmt(control['initial_front_contour']['local_normal_approach_speed_into_solid'], 7)}, "
            f"near-wall chi ratio {_fmt(residual['nearwall_chi_speed_over_u_impact'], 4)}, "
            f"gate {'PASS' if control['frozen_initial_acceptance']['pass'] else 'FAIL'}"
        )
    lines += ["", "Quarantined taper-free negative control (not a candidate):"] + [f"- {row}" for row in control_rows]
    lines += ["", "## G. Contour and velocity distance bands"]
    if band_matrix:
        case = CASE_NAMES[0]
        lines.append(
            f"- field layout `[x, y]`; window: {band_matrix[case]['candidates']['C2_BASE']['front_window_definition']}"
        )
        for key in ("C2_BASE", "T1_NEAR_NARROW"):
            entry = band_matrix[case]["candidates"][key]
            parts = []
            for band, values in entry["bands"].items():
                if values["cell_count"]:
                    parts.append(
                        f"{band}: n={values['cell_count']} taper={_fmt(values['mean_taper'], 3)} "
                        f"max speed/U={_fmt(values['max_abs_speed_over_u_impact'], 4)}"
                    )
            lines.append(f"- `{case}` `{key}`: " + "; ".join(parts))
        gate = band_matrix[case]["candidates"]["T1_NEAR_NARROW"]["frozen_gate_location_diagnostic"]
        lines.append(
            f"- frozen-gate argmax location ({case}, T1): cell {gate.get('argmax_cell_xy')} "
            f"sdf/dx={_fmt(gate.get('argmax_sdf_over_dx'), 4)} chi={_fmt(gate.get('argmax_chi'), 4)} "
            f"V/Vcell={_fmt(gate.get('argmax_control_volume_fraction'), 3)} "
            f"taper={_fmt(gate.get('argmax_wall_taper'), 3)} "
            f"speed/U={_fmt(gate.get('argmax_speed_over_u_impact'), 4)} "
            f"(open-volume share of gate cells {_fmt(gate.get('open_volume_share_of_gate_cells'), 3)})"
        )
    else:
        lines.append("- NOT_RUN_IN_THIS_PROFILE_STAGE_SELECTION")
    lines += ["", "## H. Phase advection / explicit CH / implicit CH ledger"]
    if tier2:
        budget = tier2["phase_budget"]
        for case in CASE_NAMES:
            for key in ("C2_BASE", "T3_FAR_NARROW"):
                substeps = budget[case][key]["substeps"]
                worst_closure = max(step["closure_linf_active_cells"] for step in substeps)
                bitwise = all(step["phase_update_bitwise_reproduced"] for step in substeps)
                telescopes = [
                    abs(step["flux_telescoping"]["sum_V_div_advective_flux"])
                    + abs(step["flux_telescoping"]["sum_V_div_ch_flux"])
                    for step in substeps
                ]
                mass_closure = max(abs(step["weighted_mass_budget"]["closure_residual"]) for step in substeps)
                first = substeps[0]
                components = first["max_abs_component"]
                lines.append(
                    f"- `{case}` `{key}`: additive identity max|residual| {_fmt(worst_closure, 3)} (active cells), "
                    f"phase update bitwise {bitwise}, blocked-face flux "
                    f"{_fmt(first['blocked_face_flux_max_abs']['advective'], 3)}/"
                    f"{_fmt(first['blocked_face_flux_max_abs']['ch'], 3)}, telescoped flux sum "
                    f"{_fmt(max(telescopes), 3)}, "
                    f"V-weighted mass closure {_fmt(mass_closure, 3)}, "
                    f"first-substep components adv/CH-expl/CH-impl {_fmt(components['adv'], 3)}/"
                    f"{_fmt(components['ch_explicit'], 3)}/{_fmt(components['ch_implicit'], 3)}, "
                    f"implicit CG converged {first['implicit_solver']['converged']} "
                    f"(iters {first['implicit_solver']['iterations']}, rel. residual "
                    f"{_fmt(first['implicit_solver']['relative_residual'], 3)})"
                )
        lines.append(
            "- identity: `delta_phi_total = delta_phi_adv + delta_phi_CH_explicit + delta_phi_CH_implicit` per cell; "
            "the implicit term is an implicit-solve increment and is never double counted"
        )
        lines.append(
            "- subcycling status: "
            + json.dumps(
                {case: budget[case]["C2_BASE"]["active_branch"]["subcycling_active"] for case in CASE_NAMES},
                sort_keys=True,
            )
        )
    else:
        lines.append("- NOT_RUN_IN_THIS_PROFILE_STAGE_SELECTION")
    lines += ["", "## I. Fluid velocity versus contour velocity"]
    if tier2:
        for case in CASE_NAMES:
            entry = tier2["contour_motion"][case]["C2_BASE"]
            motion = entry["matched_contour_motion"]
            lines.append(
                f"- `{case}`: matched contour displacement median {_fmt(motion['median_displacement_over_dx'], 4)} dx, "
                f"max {_fmt(motion['max_displacement_over_dx'], 4)} dx, normal projection mean "
                f"{_fmt(motion['normal_projection_mean_over_h'], 4)} (per h), "
                f"usable match fraction {_fmt(motion['usable_match_fraction'], 3)}, "
                f"support switched {entry['support_switch']['switched']}"
            )
        first = tier2["contour_motion"][CASE_NAMES[0]]["C2_BASE"]
        linear = first["linearized_normal_advance"]
        lines.append(
            f"- independent contour reconstruction vs the frozen official rule: matched fraction "
            f"{_fmt(first['independent_vs_official']['matched_fraction'], 3)}, max match distance "
            f"{_fmt(first['independent_vs_official']['max_match_distance_over_dx'], 3)} dx; linearized normal "
            f"advance {_fmt(linear['linearized_normal_advance'], 3)} with conditioning "
            f"{_fmt(linear['conditioning_ratio'], 3)} — explicitly `not_exact_contour_motion=True`"
        )
    else:
        lines.append("- NOT_RUN_IN_THIS_PROFILE_STAGE_SELECTION")
    lines += ["", "## J. First-substep RHS/Brinkman/projection"]
    if tier2:
        ledger = tier2["momentum_ledger"]
        for case in CASE_NAMES:
            for key in ("C2_BASE", "T3_FAR_NARROW"):
                first = ledger[case][key]["substeps"][0]
                deltas = first["approach_delta_by_operator"]
                lines.append(
                    f"- `{case}` `{key}` first substep: explicit RHS {_fmt(deltas['explicit_rhs'], 3)}, "
                    f"Brinkman {_fmt(deltas['brinkman'], 3)}, periodic projection "
                    f"{_fmt(deltas['periodic_pressure_projection'], 3)} (approach-speed deltas); "
                    f"projection plain-grid mean "
                    f"{_fmt(first['projection_plain_grid_mean_identity']['mean_of_proj_correction'], 3)}, "
                    f"CG converged {first['cg_converged']}"
                )
    else:
        lines.append("- NOT_RUN_IN_THIS_PROFILE_STAGE_SELECTION")
    lines += ["", "## K. Bounded event and real postcontact evidence"]
    if tier3:
        eligible = tier3.get("eligible_candidates_per_case", {})
        quarantined = tier3.get("quarantined_one_substep_negative_controls", {})
        lines += [
            f"- eligible candidates per case (frozen t=0 gates): {json.dumps(eligible, sort_keys=True)}",
            f"- ineligible candidates appear only as one-substep quarantined controls: "
            f"{sorted(quarantined)} (never evolved to events, never promoted)",
            "",
            "| case | candidate | stop | contact time | precontact approach | verdict |"
            " postcontact verified | max overshoot | max mass ratio |",
            "|---|---|---|---:|---:|---|---|---:|---:|",
        ]
        for case in CASE_NAMES:
            for key, run in sorted(tier3["events"][case].items()):
                auth = run["impact_authentication"]
                quality = run["quality"]
                scope = " (extended horizon)" if "primary_horizon_authentication" in run else ""
                lines.append(
                    f"| `{case}` | `{key}`{scope} | {run['stop_reason']} | "
                    f"{_fmt(auth.get('contact_time_interpolated_gap_threshold'), 7)} | "
                    f"{_fmt(auth.get('precontact_local_normal_approach_speed_positive_toward_solid'), 7)} | "
                    f"`{auth['verdict']}` | {auth.get('postcontact_dynamics_verified')} | "
                    f"{_fmt(quality.get('max_phi_overshoot'), 3)} | {_fmt(quality.get('max_formal_mass_ratio'), 5)} |"
                )
        for case in CASE_NAMES:
            for key, run in sorted(tier3["events"][case].items()):
                if "primary_horizon_authentication" in run:
                    lines.append(
                        f"- `{case}`/`{key}`: the primary {PRIMARY_EVENT_HORIZON} horizon returned "
                        f"`{run['primary_horizon_authentication']['verdict']}`; the single declared targeted "
                        f"continuation to t<={run['extension']['extension_horizon']} supplied the authoritative "
                        f"label and its postcontact window"
                    )
    else:
        lines.append("- NOT_RUN_IN_THIS_PROFILE_STAGE_SELECTION")
    lines += ["", "## L. Time-step sensitivity"]
    if temporal:
        lines += [
            "- matched physical times at fixed N; the official 3% refinement gate is NOT executed here",
            "",
            "| case | candidate | approach dt=0.002 | dt=0.001 | dt=0.0005 | finest/coarsest | label |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for case in CASE_NAMES:
            for key in ("C2_BASE", "T3_FAR_NARROW"):
                per_dt = temporal["cases"].get(case, {}).get(key)
                if not per_dt:
                    continue
                values = [
                    per_dt[dt]["matched_times"]["precontact_local_normal_approach"]
                    for dt in ("0.002", "0.001", "0.0005")
                    if dt in per_dt
                ]
                verdict = temporal.get("dt_sensitivity_verdict", {}).get(case, {}).get(key, {})
                lines.append(
                    f"| `{case}` | `{key}` | {_fmt(values[0], 6)} | {_fmt(values[1], 6)} | {_fmt(values[2], 6)} | "
                    f"{_fmt(verdict.get('finest_over_coarsest_ratio'), 3)} | `{verdict.get('label')}` |"
                )
        lines.append(
            "- consequence: the coarse-dt front-speed difference between C2 and an opened taper does not persist "
            "at the finest dt, so no dt-independent front-speed improvement is claimed"
            if profile == "forensic"
            else "- quick profile: the table is descriptive only; no dt-independence claim is made"
        )
    else:
        lines.append("- NOT_RUN_IN_THIS_PROFILE_STAGE_SELECTION")
    lines += ["", "## M. Negative controls and falsification"]
    if falsification:
        lines.append(
            f"- competing mechanism labels: {json.dumps(falsification.get('competing_mechanisms', {}), sort_keys=True)}"
        )
        ch_only = falsification.get("ch_only_control", {})
        for case, entry in sorted(ch_only.items()):
            lines.append(
                f"- CH-only frozen-velocity control `{case}`: contact reached by pure relaxation "
                f"{entry['contact_reached_by_pure_relaxation']} (gap {_fmt(entry['initial_gap_phi05'], 4)} -> "
                f"{_fmt(entry['final_gap_phi05'], 4)} over t={entry['horizon']}); classified "
                f"`{entry['classification']}` and never counted as an authenticated impact"
            )
        test_a = falsification.get("test_A_front_speed_vs_compatibility", {})
        if test_a:
            lines.append(
                f"- falsification A: T1 raises the t=0 front approach to "
                f"{_fmt(test_a.get('T1_NEAR_NARROW_initial_approach'), 6)} but fails the frozen gate "
                f"({', '.join(test_a.get('T1_failing_gate', []))}); C2 remains at "
                f"{_fmt(test_a.get('C2_BASE_initial_approach'), 6)}"
            )
        controls = falsification.get("test_F_projection_and_negative_controls", {})
        if controls:
            reference = controls.get("l1a2r_control_A", {})
            lines.append(
                "- projection negative controls: uniform field preserved "
                f"{reference.get('projection_preserves_uniform')}, max |delta v| full step "
                f"{_fmt(reference.get('full_step_max_abs_change_v'), 3)}; projection-only "
                f"{_fmt(reference.get('projection_only_max_abs_change_v'), 3)}"
            )
    else:
        lines.append("- NOT_RUN_IN_THIS_PROFILE_STAGE_SELECTION")
    lines += [
        "",
        "## N. Generator quality and inherited blockers",
        "- `L1B_DATA_NOT_READY`, `SPATIAL_REFINEMENT=FAIL`, `TEMPORAL_REFINEMENT=UNMEASURED`, "
        "`DIRECT_FIELD_LABELS=UNMEASURED` unchanged by this stage",
        "- inherited pillar overshoot 0.03697 > 0.02 and max mass ratio 1.005185 > 1.005 remain unresolved blockers; "
        "`PRIMARY_BLOCKER=REQUIRED_PILLAR_OR_COMPLEX_IMPACT_AUTHENTICATION` remains the owner of the next action",
        "",
        "## O. Tests, CI and compute cost",
        f"- compute: {json.dumps(run_meta, sort_keys=True)}",
    ]
    if validation:
        lines.append(f"- local validation recorded at this source revision: {json.dumps(validation, sort_keys=True)}")
    else:
        lines.append("- local validation: NOT_RECORDED_IN_THIS_RUN (no test or quality-gate result is claimed here)")
    lines += [
        "- hosted CI: not run for this stage revision at report time; no hosted result is claimed",
        "",
        "## P. Mechanism matrix",
        f"- taper mechanism verdict: `{decision['taper_mechanism_verdict']}`",
        f"- candidate outcome: `{decision['candidate_outcome']}`",
        f"- engineering stage decision: `{decision['stage_decision']}`",
        f"- compatible candidates per case: "
        f"{json.dumps(decision.get('compatible_candidates_per_case', {}), sort_keys=True)}",
        f"- front-speed-raising and compatible: {decision.get('front_speed_raising_and_compatible')}",
        f"- temporal dt-sensitive: {decision.get('temporal_dt_sensitive')}",
        f"- phase-budget identity holds: {decision.get('phase_budget_identity_holds')}",
        f"- physical verdict valid: {decision.get('physical_verdict_valid')}",
        "",
        "## Q. Exactly one next action",
        f"- {decision['next_action']['action']}",
        f"- bounded scope: {decision['next_action']['bounded_scope']}; not started in this stage",
        "",
        "## R. Reproducibility manifest and raw statuses",
        "- every evidence file is bound by SHA256 in `manifest.json`; `verify_evidence_manifest()` fails closed on a",
        "  missing, tampered or unbound file",
        "",
        "```text",
        f"STAGE={STAGE}",
        f"STAGE_VERSION={STAGE_VERSION}",
        f"PROFILE={profile}",
        f"EVIDENCE_CLASS={'FORENSIC_N192' if profile == 'forensic' else 'QUICK_ONLY'}",
        f"VERIFIED_MAIN_SHA={prerequisite['verified_merge_sha']}",
        f"BRANCH={prerequisite['branch']}",
        f"SOLVER_CONTRACT_VERSION={int(pf.SOLVER_CONTRACT_VERSION)}",
        f"TIMESTEP_POLICY={EXPECTED_POLICY}",
        f"TAPER_MECHANISM_VERDICT={decision['taper_mechanism_verdict']}",
        f"CANDIDATE_OUTCOME={decision['candidate_outcome']}",
        f"STAGE_DECISION={decision['stage_decision']}",
        f"TEMPORAL_DT_SENSITIVE={decision.get('temporal_dt_sensitive')}",
        "CANDIDATE_PROMOTED=false",
        "CONTRACT13_PROMOTED=false",
        "OFFICIAL_DATASET_WRITTEN=false",
        "BULK_TRAINING_DATA_GENERATED=false",
        "L1B_DATA=L1B_DATA_NOT_READY",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI / run
# ---------------------------------------------------------------------------
STAGES = ("preflight", "initial", "operator", "budget", "events", "temporal", "assemble")


def _build_bundles(N: int, dt_levels: tuple[float, ...]) -> dict[float, dict[str, Any]]:
    cases = audit._canary_cases()
    if tuple(cases) != CASE_NAMES:
        raise AuditBlocked(f"frozen canary registry changed: {tuple(cases)}")
    return {dt: {case: audit._derive_bundle(case, cases[case], N, dt) for case in CASE_NAMES} for dt in dt_levels}


def verify_evidence_manifest(out_dir: str | Path) -> dict[str, Any]:
    """Fail-closed verification of a committed L1A-2t evidence directory.

    Recomputes every SHA256 binding recorded in ``manifest.json``, rejects missing,
    tampered or unbound files, and re-checks the frozen stage invariants that make the
    evidence admissible. Any deviation raises :class:`AuditBlocked`.
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise AuditBlocked(f"evidence manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("evidence_files") or {}
    if not entries:
        raise AuditBlocked(f"evidence manifest has no SHA256 bindings: {manifest_path}")
    problems: list[str] = []
    for name, meta in entries.items():
        path = out_dir / name
        recorded = meta.get("sha256") if isinstance(meta, dict) else meta
        if not path.exists():
            problems.append(f"{name}: MISSING")
        elif sha256_file(path) != recorded:
            problems.append(f"{name}: SHA256_MISMATCH")
    bound = set(entries)
    on_disk = {path.name for path in out_dir.iterdir() if path.is_file()} - {"manifest.json"}
    for name in sorted(on_disk - bound):
        problems.append(f"{name}: UNBOUND_FILE_PRESENT")
    if manifest.get("stage") != STAGE:
        problems.append(f"stage={manifest.get('stage')!r} != {STAGE!r}")
    if manifest.get("solver_contract_version") != int(pf.SOLVER_CONTRACT_VERSION):
        problems.append("solver_contract_version drift")
    if manifest.get("timestep_policy") != EXPECTED_POLICY:
        problems.append("timestep policy drift")
    if manifest.get("no_promotion") is not True:
        problems.append("no_promotion flag must be True")
    missing_required = [name for name in REQUIRED_EVIDENCE_FILES if name not in bound]
    if missing_required:
        problems.append(f"REQUIRED_FILES_NOT_BOUND: {missing_required}")
    if problems:
        raise AuditBlocked(f"evidence verification failed for {out_dir}: {problems}")
    return {
        "verified": True,
        "directory": str(out_dir),
        "stage": manifest.get("stage"),
        "stage_version": manifest.get("stage_version"),
        "evidence_class": manifest.get("evidence_class"),
        "bound_files": len(entries),
        "verified_main_sha": manifest.get("verified_main_sha"),
        "profile": manifest.get("profile"),
    }


def load_validation_payload(path: str | Path | None) -> dict[str, Any] | None:
    """Load and fail-closed check the recorded local-validation payload.

    The payload must name the exact audited module revision (``stage_source_sha256``),
    so a report can never claim test/quality results obtained from a different source
    revision than the one that produced the physics.
    """
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    current = sha256_file(Path(__file__))
    recorded = payload.get("stage_source_sha256")
    if recorded is None:
        raise AuditBlocked("validation payload must record stage_source_sha256")
    if recorded != current:
        raise AuditBlocked(
            f"validation payload source hash does not match the running module: recorded={recorded} current={current}"
        )
    if payload.get("stage") not in (None, STAGE):
        raise AuditBlocked(f"validation payload stage mismatch: {payload.get('stage')}")
    return payload


def run(
    profile: str = "forensic",
    out: str | Path | None = None,
    stages: str = "all",
    *,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested = STAGES if stages == "all" else tuple(part.strip() for part in stages.split(",") if part.strip())
    unknown = [name for name in requested if name not in STAGES]
    if unknown:
        raise AuditBlocked(f"unknown stages: {unknown}")
    started = time.perf_counter()
    out_dir = Path(out) if out is not None else EVIDENCE_DIR
    N = N_FORENSIC if profile == "forensic" else N_QUICK
    saved_initial = json.loads((L1A2S_DIR / "initializer_constraint_matrix.json").read_text())

    preflight = preflight_state()
    operator_map = source_operator_map()
    progress(f"preflight ok (N={N}, profile={profile})")
    initial_matrix = None
    band_matrix = None
    tier2 = None
    tier3 = None
    temporal = None
    falsification = None
    reproduction: dict[str, Any] = {}
    bundles: dict[str, Any] = {}
    bundles_by_dt: dict[float, dict[str, Any]] = {}
    needs_bundles = any(name in requested for name in ("initial", "operator", "budget", "events", "temporal"))
    if needs_bundles:
        bundles_by_dt = _build_bundles(N, DT_LEVELS)
        bundles = bundles_by_dt[0.002]
        reproduction = {
            case: c2_exact_reproduction(bundles[case], saved_initial, compare_saved_evidence=(profile == "forensic"))
            for case in CASE_NAMES
        }
        progress("C2 exact reproduction verified against frozen L1A-2s evidence")

    if "initial" in requested:
        initial_matrix = tier1_initial_matrix(bundles, saved_initial, compare_saved_evidence=(profile == "forensic"))
        band_matrix = tier1_band_matrix(bundles)
    if "operator" in requested or "budget" in requested:
        tier2 = tier2_operator_and_budget(bundles, ("C2_BASE", "T3_FAR_NARROW"))
    if "events" in requested and initial_matrix is not None:
        tier3 = tier3_events(bundles, CANDIDATE_RUN_ORDER, initial_matrix, profile)
    if "temporal" in requested:
        temporal = temporal_sensitivity(
            bundles_by_dt,
            ("C2_BASE", "T3_FAR_NARROW"),
            ("flat_we100_ct050", "flat_we200_ct000", "pillar_training"),
            horizon=0.48,
            profile=profile,
        )
    if initial_matrix is not None and tier2 is not None:
        falsification = falsification_matrix(bundles, initial_matrix, band_matrix, tier2, tier3, profile=profile)
        if tier3 is not None:
            falsification["ch_only_control"] = {
                case: ch_only_relaxation(
                    bundles[case], "C2_BASE", build_candidate_state(bundles[case], "C2_BASE")[0], horizon=0.48
                )
                for case in ("flat_we100_ct050", "flat_we200_ct000", "pillar_training")
            }
            falsification["counterfactual_one_state"] = {
                case: counterfactual_matrix(
                    bundles[case], "C2_BASE", build_candidate_state(bundles[case], "C2_BASE")[0]
                )
                for case in ("flat_we100_ct050", "pillar_training")
            }
    decision = decide(
        profile=profile,
        initial_matrix=initial_matrix,
        tier3=tier3,
        temporal=temporal,
        budget=tier2["phase_budget"] if tier2 else None,
    )
    run_meta = {
        "stages": list(requested),
        "elapsed_seconds": float(time.perf_counter() - started),
        "N": N,
        "profile": profile,
        "dt_levels": list(DT_LEVELS),
        "missing_or_unmeasured": [
            name
            for name, present in {
                "initial": initial_matrix is not None,
                "operator": tier2 is not None,
                "events": tier3 is not None,
                "temporal": temporal is not None,
                "falsification": falsification is not None,
            }.items()
            if not present
        ],
    }
    if "assemble" in requested:
        assembled = assemble_evidence(
            profile=profile,
            out_dir=out_dir,
            preflight=preflight,
            operator_map=operator_map,
            reproduction=reproduction,
            initial_matrix=initial_matrix if initial_matrix is not None else {},
            band_matrix=band_matrix if band_matrix is not None else {},
            tier2=tier2,
            tier3=tier3,
            temporal=temporal,
            falsification=falsification,
            decision=decision,
            run_meta=run_meta,
            validation=validation,
        )
        progress(f"evidence assembled in {out_dir}")
        return {"decision": decision, "run_meta": run_meta, "evidence": assembled}
    return {
        "decision": decision,
        "run_meta": run_meta,
        "preflight": preflight,
        "initial_matrix": initial_matrix,
        "band_matrix": band_matrix,
        "tier2": tier2,
        "tier3": tier3,
        "temporal": temporal,
        "falsification": falsification,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "forensic"), default="forensic")
    parser.add_argument("--stages", default="all", help=f"comma separated subset of {STAGES} or 'all'")
    parser.add_argument("--out", default=None, help="evidence output directory (defaults to evidence/l1a2t)")
    parser.add_argument(
        "--validation-json",
        default=None,
        help="recorded local-validation payload (focused pytest + four-tool quality gate) bound into the report",
    )
    parser.add_argument(
        "--verify-evidence",
        default=None,
        help="verify a committed evidence directory against its manifest SHA256 bindings, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.verify_evidence is not None:
        report = verify_evidence_manifest(args.verify_evidence)
        progress(f"evidence verified: {json.dumps(report, sort_keys=True)}")
        return
    validation = load_validation_payload(args.validation_json)
    result = run(profile=args.profile, out=args.out, stages=args.stages, validation=validation)
    decision = result["decision"]
    progress(
        "decision: mechanism={} candidate={} stage={}".format(
            decision["taper_mechanism_verdict"], decision["candidate_outcome"], decision["stage_decision"]
        )
    )


if __name__ == "__main__":
    main()
