"""L1A-2y - reconstruction-stable early-window interface transport attribution.

This stage reopens the question L1A-2x had to leave open. It measures the early normal motion of the
``pillar_training`` C2 interface over the identical window (one case, ``dt = 0.002``, at most 40 macro
steps, ``t <= 0.08``, frozen contract-12 solver, frozen C2 initialization, no control arm, no
refinement, no extension toward contact) with a readout whose support is the *reconstructed contour
branch* instead of one pinned grid edge, and it issues a mechanism verdict only if every prespecified
gate passes: otherwise the verdict is withheld and every failing quantity is recorded with the numbers
that produced it.

Phase 1 (design, no CFD) is a deliverable in its own right: the readout is prespecified, its rules and
constants are frozen, it is validated on synthetic translations, a shared-grid-node crossing, two-root
ambiguity, a mirror-symmetric contour, a vanishing branch and a saddle cell - and it is checked to
reproduce the frozen reconstruction bit-for-bit on the ``t = 0`` state before the window is advanced.
The synthetic branch model also exercises the exact phenomenon that stopped L1A-2x (a crossing that
moves from one grid edge to the next is continuous) and the test suite reproduces both behaviours on the
same model field.

The comparison error of L1A-2x is re-audited here rather than inherited: the raw margin spread of its
three prespecified attribution rules is recorded as measured (about 1.5e-3 dx) together with the
reconstruction, nonlinearity and rule-to-rule terms separately, and it is *not* normalised away. The
verdict rule of this stage uses raw, dimensional errors only.

Nothing frozen is edited: no production solver, no threshold, no historical evidence file, no quality
baseline, no solver contract. The one trajectory of this stage is the only trajectory it runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np
import phasefield as pf  # noqa: E402
from production import frozen_chns_early_window_transport_attribution_audit as stage_2x  # noqa: E402
from production import impact_initialization_compatibility_audit as audit  # noqa: E402
from production import l1a2y_contour_branch_readout as readout  # noqa: E402
from production import radial_incident_seed_eta_dt_audit as stage_2u  # noqa: E402

STAGE = "L1A-2y"
STAGE_VERSION = "l1a2y_v1"
BASELINE_HEAD_SHA = "3695282e1de6c990514636b28d185ba3f4427a5b"
VERIFIED_MAIN_SHA = "09e2993b8da478d79e768054194201af57121784"
BRANCH = "arena/02b6c71b-hydrogym"
CASE_NAME = "pillar_training"
PROFILE_N = 192
REQUESTED_DT = 0.002
MAX_MACRO_STEPS = 40  # owner-mandated window cap: t <= 0.08
SUBSTEPS_PER_MACRO = 3  # frozen pf.step contract
CONTACT_GAP_THRESHOLD_DX = 1.5  # frozen event rule, used only as a stop guard, never scored

# --- prespecified decision constants (declared before the window is measured) ----------------------
DOMINANCE_FACTOR = 3.0
ATTRIBUTION_CLOSURE_BAND_DX = 0.01
TOTAL_DISPLACEMENT_CLOSURE_BAND_DX = 0.01
EQUIVALENCE_MAX_PHI_DIFF = 1e-6
EQUIVALENCE_MAX_GAP_DIFF_DX = 1e-4
WALK_NOISE_FLOOR_DX = 1.0e-3
CROSS_CHECK_TOLERANCE_DX = 0.05
STATIC_RECONSTRUCTION_TOLERANCE_DX = 0.01

DECISION_STATES = (
    "ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION",
    "CAHN_HILLIARD_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION",
    "MIXED_TERMS_WITHIN_THE_COMPARISON_ERROR",
    "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED",
)

FROZEN_STAGES = (
    ("L1A-2t", "l1a2t"),
    ("L1A-2u", "l1a2u"),
    ("L1A-2v", "l1a2v"),
    ("L1A-2w", "l1a2w"),
    ("L1A-2x", "l1a2x"),
)

EVIDENCE_FILES = (
    "preflight.json",
    "frozen_evidence_verification.json",
    "source_and_provenance.json",
    "readout_design.json",
    "synthetic_reader_validation.json",
    "frozen_reconstruction_equivalence.json",
    "comparison_error_reaudit.json",
    "initial_state_identity.json",
    "trajectory_specification.json",
    "substep_ledger.json",
    "interface_attribution.json",
    "static_reconstruction_study.json",
    "residual_decomposition.json",
    "conservation_and_convergence.json",
    "capture_equivalence.json",
    "validity_screen.json",
    "mechanism_decision.json",
    "adversarial_and_falsification.json",
    "quality_and_blocker_matrix.json",
    "next_action.json",
    "l1a2y_final_report.json",
    "l1a2y_final_report.md",
    "manifest.json",
)

STAGE_OWNED_PATHS = (
    "examples/two_phase/production/l1a2y_contour_branch_readout.py",
    "examples/two_phase/production/frozen_reconstruction_branch_attribution_audit.py",
    "examples/two_phase/tests/test_frozen_reconstruction_branch_attribution_audit.py",
    "examples/two_phase/evidence/l1a2y",
)

TWO_PHASE = Path(__file__).resolve().parents[1]
REPO_ROOT = TWO_PHASE.parents[1]
EVIDENCE_OWN = TWO_PHASE / "evidence" / "l1a2y"
READOUT_MODULE = TWO_PHASE / "production" / "l1a2y_contour_branch_readout.py"

FROZEN_BLOCKED = (AuditBlocked := stage_2u.AuditBlocked, audit.AuditValidationError)


# ---------------------------------------------------------------------------
# Section A - helpers, provenance and fail-closed preflight
# ---------------------------------------------------------------------------
def progress(message: str) -> None:
    print(f"[{STAGE}] {message}", flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _git(args: list[str], *, required: bool = True) -> str | None:
    completed = subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        if required:
            raise AuditBlocked(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
        return None
    return completed.stdout.strip()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha(array: Any) -> str:
    """Array hash through the frozen audit implementation, so it matches the sealed evidence."""
    return audit._array_sha(np.asarray(array))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _porcelain_path(line: str) -> str:
    path = line[2:] if len(line) > 2 else ""
    return path.strip().strip('"')


def _stage_owned(path: str) -> bool:
    return any(path == owned or path.startswith(f"{owned}/") for owned in STAGE_OWNED_PATHS)


def _out_dir(out: str | Path | None) -> Path:
    return Path(out) if out is not None else EVIDENCE_OWN


def preflight_state() -> dict[str, Any]:
    """Fail-closed preflight: branch, head, lineage and worktree hygiene."""
    head = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    porcelain = _git(["status", "--porcelain"]) or ""
    dirty = [_porcelain_path(line) for line in porcelain.splitlines() if line.strip()]
    foreign = [path for path in dirty if not _stage_owned(path)]
    history = _git(["log", "--format=%H", f"{BASELINE_HEAD_SHA}~1..{BASELINE_HEAD_SHA}"], required=False)
    if not history:
        raise AuditBlocked(f"baseline head {BASELINE_HEAD_SHA} is not in this repository")
    commits = (_git(["log", "--format=%H", f"{BASELINE_HEAD_SHA}..HEAD"]) or "").split()
    unknown = [commit for commit in commits if not _commit_touches_only_stage_paths(commit)]
    payload = {
        "stage": STAGE,
        "branch": branch,
        "head_sha": head,
        "baseline_head_sha": BASELINE_HEAD_SHA,
        "baseline_pull_request": 24,
        "baseline_rule": (
            f"this stage ran from the L1A-2x evidence head {BASELINE_HEAD_SHA}; every commit above it "
            "must be stage owned or the stage refuses to measure"
        ),
        "main_sha": VERIFIED_MAIN_SHA,
        "worktree_clean_outside_stage_deliverables": not foreign,
        "dirty_paths_outside_stage_deliverables": foreign,
        "stage_owned_dirty_paths": [path for path in dirty if _stage_owned(path)],
        "commits_above_baseline": commits,
        "commits_above_baseline_not_stage_owned": unknown,
        "shallow_repository": (_git(["rev-parse", "--is-shallow-repository"]) or "unknown") == "true",
        "stage_evidence_tracked_at_head": bool(
            (_git(["ls-files", "examples/two_phase/evidence/l1a2y"], required=False) or "").strip()
        ),
    }
    if branch != BRANCH:
        raise AuditBlocked(f"not on the sanctioned branch: {branch!r}")
    if payload["shallow_repository"]:
        raise AuditBlocked("a shallow clone cannot verify the frozen lineage; refusing to measure")
    if foreign:
        raise AuditBlocked(f"worktree dirty outside the stage deliverables: {foreign}")
    if unknown:
        raise AuditBlocked(f"commits above the baseline are not stage owned: {unknown}")
    own_manifest = EVIDENCE_OWN / "manifest.json"
    evidence_committed_but_absent = (
        payload["stage_evidence_tracked_at_head"] and not own_manifest.is_file() and not dirty
    )
    if evidence_committed_but_absent:
        raise AuditBlocked(
            "this head has committed stage evidence but the evidence directory is missing from the worktree; "
            "refusing to measure, because regenerating it here would silently repair a committed deletion"
        )
    return payload


def _commit_touches_only_stage_paths(commit: str) -> bool:
    names = _git(["show", "--name-only", "--format=", commit]) or ""
    paths = [path.strip() for path in names.splitlines() if path.strip()]
    return bool(paths) and all(_stage_owned(path) for path in paths)


def verify_prerequisites() -> dict[str, Any]:
    """Re-hash the sealed evidence of L1A-2t/u/v/w/x and re-verify every recorded source hash."""
    out: dict[str, Any] = {"stages": {}, "failures": [], "verified_main_sha": VERIFIED_MAIN_SHA}
    for stage_name, directory in FROZEN_STAGES:
        path = TWO_PHASE / "evidence" / directory
        manifest_path = path / "manifest.json"
        entry: dict[str, Any] = {"stage": stage_name, "directory": f"evidence/{directory}"}
        if not manifest_path.is_file():
            entry["status"] = "MISSING_MANIFEST"
            out["failures"].append(f"{stage_name}: missing manifest.json")
            out["stages"][stage_name] = entry
            continue
        manifest = json.loads(manifest_path.read_text())
        problems: list[str] = []
        if manifest.get("stage") != stage_name:
            problems.append(f"stage field is {manifest.get('stage')!r}")
        if manifest.get("verified_main_sha") != VERIFIED_MAIN_SHA:
            problems.append(f"verified main sha is {manifest.get('verified_main_sha')!r}")
        seal = stage_2x._recorded_evidence_seal(manifest)
        if not seal:
            problems.append("manifest declares no bound-file seal under files or evidence_files")
        for name, want in sorted(seal.items()):
            target = path / name
            if not target.is_file():
                problems.append(f"missing bound file {name}")
            elif sha256_file(target) != want:
                problems.append(f"sha mismatch {name}")
        sources = stage_2x._verify_recorded_sources(manifest.get("source_hashes"))
        problems.extend(sources["problems"])
        entry.update(
            {
                "status": "VERIFIED" if not problems else "MISMATCH",
                "manifest_sha256": sha256_file(manifest_path),
                "bound_files": len(seal),
                "verdict": manifest.get("verdict"),
                "recorded_hosted_ci": (manifest.get("hosted_ci_reseal") or {}).get("status"),
                "recorded_baseline_head": manifest.get("baseline_head_sha"),
                "source_hashes_recorded": sources["checked"],
                "source_hashes_verified_from_the_worktree": len(sources["verified_from_the_worktree"]),
                "source_hashes_verified_from_commit_history": len(sources["verified_from_commit_history"]),
                "source_hashes_unverifiable": sources["unverifiable"],
                "source_hash_provenance": {
                    "from_the_worktree": sources["verified_from_the_worktree"],
                    "from_commit_history": sources["verified_from_commit_history"],
                },
                "problems": problems,
            }
        )
        if problems:
            out["failures"].append(f"{stage_name}: {problems}")
        out["stages"][stage_name] = entry
    out["all_verified"] = not out["failures"]
    out["rule"] = (
        "every prerequisite evidence file is re-hashed from disk against the seal of its own manifest "
        "(both frozen seal conventions are read), every source hash those manifests recorded is "
        "re-verified by content against the worktree or against the commit history of its own path, and a "
        "manifest that declares no recognizable seal is refused instead of counted as verified; a single "
        "mismatch blocks the stage before any solver call"
    )
    if out["failures"]:
        raise AuditBlocked(f"frozen prerequisite evidence failed verification: {out['failures']}")
    return out


def source_and_provenance(preflight: dict[str, Any]) -> dict[str, Any]:
    """Record what ran, from where, and which frozen sources it depends on."""
    frozen_sources = {
        "examples/two_phase/phasefield.py": sha256_file(TWO_PHASE / "phasefield.py"),
        "examples/two_phase/production/impact_initialization_compatibility_audit.py": sha256_file(
            TWO_PHASE / "production" / "impact_initialization_compatibility_audit.py"
        ),
        "examples/two_phase/production/radial_incident_seed_eta_dt_audit.py": sha256_file(
            TWO_PHASE / "production" / "radial_incident_seed_eta_dt_audit.py"
        ),
        "examples/two_phase/production/frozen_chns_early_window_transport_attribution_audit.py": sha256_file(
            TWO_PHASE / "production" / "frozen_chns_early_window_transport_attribution_audit.py"
        ),
    }
    return {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "branch": preflight["branch"],
        "head_sha": preflight["head_sha"],
        "baseline_head_sha": BASELINE_HEAD_SHA,
        "main_sha": VERIFIED_MAIN_SHA,
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "phase_storage_model": str(pf.PHASE_STORAGE_MODEL),
        "phase_boundary_model": str(pf.PHASE_BOUNDARY_MODELS[0]),
        "phase_boundary_models_available": [str(model) for model in pf.PHASE_BOUNDARY_MODELS],
        "phase_transport_geometry": str(pf.PHASE_TRANSPORT_GEOMETRY),
        "module_path": _display_path(Path(__file__)),
        "module_sha256": sha256_file(__file__),
        "readout_module_path": _display_path(READOUT_MODULE),
        "readout_module_sha256": sha256_file(READOUT_MODULE),
        "frozen_status_flags": {
            "contract": int(pf.SOLVER_CONTRACT_VERSION),
            "L1A_STATUS": "BLOCKED",
            "L1B_DATA_NOT_READY": "unchanged",
        },
        "frozen_source_sha256": frozen_sources,
        "trajectories_run": 1,
        "trajectory_definition": (
            "exactly one case (pillar_training), one dt (0.002), one initial state (frozen L1A-2s C2), at "
            "most 40 frozen macro steps; no control arm, no refinement, no extension toward contact"
        ),
        "read_only_study_inputs": [
            "production/frozen_chns_early_window_transport_attribution_audit.py (L1A-2x): its frozen-kernel "
            "substep capture, its fixed support probe and its evidence/seal helpers, imported and never edited",
            "evidence/l1a2x: the sealed L1A-2x payloads, read for the comparison-error re-audit and for the "
            "cross-check of this stage's identical window",
        ],
    }


def _initial_state_identity() -> dict[str, Any]:
    """Bind the frozen C2 initial state to the seals recorded by the upstream stages."""
    cases = audit._canary_cases()
    if CASE_NAME not in cases:
        raise AuditBlocked(f"frozen canary {CASE_NAME!r} is not available")
    bundle = audit._derive_bundle(CASE_NAME, cases[CASE_NAME], PROFILE_N, REQUESTED_DT)
    state0, details = stage_2u.build_any_state(bundle, stage_2u.C2_KEY)
    if "radial" in str(details.get("source", "")).lower():
        raise AuditBlocked("the initial state is a radial seed; this stage mandates the frozen C2 state")
    p = bundle.p
    support = stage_2x.fixed_support(bundle, state0)
    sealed = json.loads((TWO_PHASE / "evidence" / "l1a2w" / "contour_normal_velocity_and_flux.json").read_text())
    sealed_candidate = sealed["cases"][CASE_NAME]["candidates"][stage_2u.C2_KEY]
    sealed_hashes: dict[str, Any] = {
        "phi": sealed_candidate["phi_hash"],
        "u": sealed_candidate["u_hash"],
        "v": sealed_candidate["v_hash"],
    }
    x2 = json.loads((TWO_PHASE / "evidence" / "l1a2x" / "initial_state_identity.json").read_text())
    # the frozen evidence hashes the state arrays as they are stored (phase float64, velocity legacy
    # precision), so the hashes are taken from the arrays themselves and never from a cast copy
    hashes = {name: _array_sha(getattr(state0, name)) for name in ("phi", "u", "v")}
    parameters = {
        "N": int(p.Nx),
        "dx": float(p.dx),
        "dy": float(p.dy),
        "Lx": float(p.Lx),
        "Ly": float(p.Ly),
        "eps": float(p.eps),
        "M": float(p.M),
        "We": float(p.We),
        "Re": float(p.Re),
        "eta_pen": float(p.eta_pen),
        "rho_l": float(p.rho_l),
        "rho_g": float(p.rho_g),
        "requested_dt": REQUESTED_DT,
        "effective_dt": float(p.dt),
        "substep_dt": float(p.dt) / SUBSTEPS_PER_MACRO,
        "ch_solver_rtol": float(p.ch_solver_rtol),
        "ch_solver_max_iterations": int(p.ch_solver_max_iterations),
        "u_impact": float(bundle.u_impact),
        "initial_gap": float(bundle.initial_gap),
        "local_surface_top": float(bundle.local_surface_top),
        "cos_theta": float(cases[CASE_NAME].get("cos_theta", 0.0)),
    }
    solid = bundle.solid
    geometry = {
        "solid_area_fraction_chi_ge_0p01": float(np.mean(np.asarray(solid.chi, dtype=np.float64) >= 0.01)),
        "open_volume_total": float(np.sum(np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64))),
        "closed_face_count_x": int(np.sum(np.asarray(pf.phase_face_apertures(solid, p)[0]) == 0.0)),
        "closed_face_count_y": int(np.sum(np.asarray(pf.phase_face_apertures(solid, p)[1]) == 0.0)),
    }
    return {
        "stage": STAGE,
        "case_name": CASE_NAME,
        "profile_N": PROFILE_N,
        "state_source": "frozen audit.build_candidate SDF_TAPERED_STREAMFUNCTION_V1 (L1A-2s C2)",
        "candidate_key": stage_2u.C2_KEY,
        "hashes": hashes,
        "l1a2x_sealed_hashes": x2["hashes"],
        "identity_checks": {
            "phi_hash": hashes["phi"] == x2["hashes"]["phi"],
            "u_hash": hashes["u"] == x2["hashes"]["u"],
            "v_hash": hashes["v"] == x2["hashes"]["v"],
            "sealed_l1a2w_phi_hash": hashes["phi"] == sealed_candidate["phi_hash"],
            "sealed_l1a2w_u_hash": hashes["u"] == sealed_candidate["u_hash"],
            "sealed_l1a2w_v_hash": hashes["v"] == sealed_candidate["v_hash"],
            "sealed_l1a2w_contour_cell": list(support["cell"]) == list(sealed_candidate["contour_support_cell"]),
            "sealed_l1a2w_contour_cell_chi": math.isclose(
                float(support["chi_at_cell"]), float(sealed_candidate["contour_cell_chi"]), rel_tol=0.0, abs_tol=1e-12
            ),
            "sealed_l1a2w_normal_velocity": math.isclose(
                stage_2x._sample(state0.v, tuple(support["point"]), bundle) / float(bundle.u_impact),
                float(sealed_candidate["contour_cell_normal_velocity_over_u_impact"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "state_t_zero": float(np.asarray(state0.t)) == 0.0,
        },
        "support": support,
        "parameters": parameters,
        "geometry": geometry,
        "parameter_hash": hashlib.sha256(
            json.dumps(parameters, sort_keys=True, default=_json_default).encode("utf-8")
        ).hexdigest(),
        "geometry_hash": hashlib.sha256(
            json.dumps(geometry, sort_keys=True, default=_json_default).encode("utf-8")
        ).hexdigest(),
        "sealed_hashes": x2["hashes"],
        "sealed_l1a2w_candidate_hashes": sealed_hashes,
        "sealed_l1a2w_probe_cases": sorted(sealed["cases"]),
        "prohibited_alternatives_not_used": [
            "UNIFORM_ALL_DOMAIN production initializer",
            "STREAMFUNCTION_LOCALIZED_V0 replay control",
            "any L1A-2w exploratory ramp construction",
        ],
    }


def initial_state_identity() -> dict[str, Any]:
    payload = _initial_state_identity()
    if not all(payload["identity_checks"].values()):
        raise AuditBlocked(f"the frozen C2 initial state does not reproduce the sealed identity: {payload}")
    return payload


# ---------------------------------------------------------------------------
# Section B - phase 1: readout design and its validation (no CFD)
# ---------------------------------------------------------------------------
def readout_design(identity: dict[str, Any]) -> dict[str, Any]:
    """Pin the tracked branch and freeze the design before the window is measured."""
    bundle, state0 = stage_2x._bundle_and_state()
    p = bundle.p
    spec = readout.branch_definition(p, np.asarray(state0.phi, dtype=np.float64), identity["support"])
    design = readout.design_payload(spec)
    t0 = readout.branch_root(np.asarray(state0.phi, dtype=np.float64), spec, spec["t0_root_over_dx"], with_arc=True)
    if t0["status"] != "OK":
        raise AuditBlocked(f"the pinned branch is not resolvable on the t = 0 state: {t0['status']}")
    pinned_cell = identity["support"]["cell"]
    row = np.asarray(state0.phi, dtype=np.float64)[pinned_cell[0], pinned_cell[1]]
    design.update(
        {
            "stage": STAGE,
            "pinned_branch_model": {
                "field": "f(y) = 0.45 * tanh((y - a) / eps_num) - 0.025 * tanh((y - b) / eps_num), dy = 0.1 dx",
                "crossings_at_t0_over_dx": [
                    spec["t0_root_over_dx"] - 4.0,
                    spec["t0_root_over_dx"],
                    spec["t0_root_over_dx"] + 4.0,
                ],
                "matched_t0_root_over_dx": spec["t0_root_over_dx"],
                "measured_at_the_pinned_column_over_dx": spec["column_x_over_dx"],
                "rule": (
                    "the synthetic model is built so that the middle crossing sits exactly on a shared grid "
                    "node at t = 0, which is the support configuration L1A-2x met at its macro step 25; the "
                    "same model field is then translated and the readout must report it without ambiguity"
                ),
            },
            "t0_readout": t0,
            "t0_pinned_point_phi_level": float(row),
            "measured_quantity": (
                "the displacement, in dx units, of the tracked phi = 0.5 branch along the fixed wall-normal "
                "ray, positive toward the solid; every ingredient displacement is the same functional "
                "applied to the field that its own increment produces"
            ),
            "verdict_rule": {
                "dominance_factor": DOMINANCE_FACTOR,
                "attribution_closure_band_dx": ATTRIBUTION_CLOSURE_BAND_DX,
                "total_displacement_closure_band_dx": TOTAL_DISPLACEMENT_CLOSURE_BAND_DX,
                "walk_noise_floor_dx": WALK_NOISE_FLOOR_DX,
                "cross_check_tolerance_dx": CROSS_CHECK_TOLERANCE_DX,
                "statement": (
                    "margin = |d_adv| - |d_CH_expl + d_CH_impl| compared against DOMINANCE_FACTOR times a "
                    "comparison error made of raw dimensional terms only: the spread of the three "
                    "prespecified attribution rules, the measured walk-vs-single-step noise floor, and the "
                    "arc/root agreement term. No term is normalised by the displacement it measures. The "
                    "verdict is withheld whenever any validity guard fails"
                ),
            },
        }
    )
    return design


def synthetic_reader_validation() -> dict[str, Any]:
    """Phase-1 adversarial validation of the readout: synthetic fields, no solver, no evidence read."""
    report = readout.synthetic_validation_report()
    report["stage"] = STAGE
    report["checks_pass"] = bool(report["all_hold"])
    if not report["all_hold"]:
        raise AuditBlocked("the synthetic reader validation failed; refusing to measure")
    return report


def frozen_reconstruction_equivalence(identity: dict[str, Any]) -> dict[str, Any]:
    """Show that the readout reproduces the frozen reconstruction on the t = 0 state, bit for bit."""
    bundle, state0 = stage_2x._bundle_and_state()
    p = bundle.p
    phi = np.asarray(state0.phi, dtype=np.float64)
    rng = np.random.default_rng(20261012)
    xs = rng.uniform(0.0, float(p.Lx), 512)
    ys = rng.uniform(0.0, float(p.Ly), 512)
    frozen_values = np.asarray(audit._periodic_bilinear_sample(phi, xs, ys, p), dtype=np.float64)
    readout_values = np.asarray(readout.bilinear_sample(phi, xs, ys, p), dtype=np.float64)
    sampler_difference = float(np.max(np.abs(frozen_values - readout_values)))
    contour = audit._threshold_contour_edge_crossings(phi, 0.5, p)
    column = float(identity["support"]["point"][0]) / float(p.dx)
    frozen_column = np.sort(contour[np.abs(contour[:, 0] / float(p.dx) - column) < 1.0e-9][:, 1] / float(p.dy))
    readout_column = np.sort(
        np.array(
            [
                root["y_over_dx"]
                for root in readout.frozen_column_crossings(phi, round(column - 0.5), p)
                if abs(root["y_over_dx"] - float(identity["support"]["point"][1]) / float(p.dy))
                <= readout.RAY_SEARCH_HALF_WINDOW_DX
            ]
        )
    )
    support = identity["support"]
    edge_root = float(support["gap_over_dx"])  # the frozen global rule's own gap is a different quantity
    spec = readout.branch_definition(p, phi, support)
    checks = {
        "sampler_matches_the_frozen_sampler": sampler_difference <= 1.0e-14,
        "column_crossings_match_the_frozen_reconstruction": bool(
            readout_column.size == 1
            and abs(readout_column[0] - float(identity["support"]["point"][1]) / float(p.dy)) <= 1.0e-12
        ),
        "pinned_root_is_the_frozen_crossing": abs(spec["t0_root_over_dx"] - support["point"][1] / float(p.dy))
        <= 1.0e-12,
        "pinned_segment_is_the_one_l1a2x_tracked": spec["t0_root_segment_index"] == 23,
        "pinned_column_is_the_frozen_support_column": spec["column_index"] == int(support["cell"][0]),
    }
    payload = {
        "stage": STAGE,
        "frozen_sampler": "impact_initialization_compatibility_audit._periodic_bilinear_sample",
        "frozen_contour": "impact_initialization_compatibility_audit._threshold_contour_edge_crossings",
        "sampler_max_abs_difference": sampler_difference,
        "frozen_column_crossings_over_dx": [float(value) for value in frozen_column],
        "readout_column_crossings_over_dx": [float(value) for value in readout_column],
        "pinned_branch": {
            "column_index": spec["column_index"],
            "column_x_over_dx": spec["column_x_over_dx"],
            "t0_root_over_dx": spec["t0_root_over_dx"],
            "t0_root_segment_index": spec["t0_root_segment_index"],
            "t0_root_at_shared_node": spec["t0_root_at_shared_node"],
        },
        "support_gap_over_dx_reference": edge_root,
        "checks": checks,
        "all_hold": all(checks.values()),
        "rule": (
            "the readout is only used if it reproduces the frozen sampler and the frozen linear contour "
            "reconstruction on the actual t = 0 field, and if the branch it pins is the very segment L1A-2x "
            "tracked"
        ),
    }
    if not payload["all_hold"]:
        raise AuditBlocked(f"the readout does not reproduce the frozen reconstruction: {checks}")
    return payload


def comparison_error_reaudit() -> dict[str, Any]:
    """Re-audit the L1A-2x comparison error from its sealed evidence, without normalising it away."""
    x2 = json.loads((TWO_PHASE / "evidence" / "l1a2x" / "interface_attribution.json").read_text())
    family = x2["family"]
    rules = {name: float(entry["margin_dx"]) for name, entry in family.items()}
    primary = "PRIMARY_POINT_T0_COEFFICIENT"
    alternatives = [name for name in rules if name != primary]
    raw_spread = max(abs(rules[name] - rules[primary]) for name in alternatives)
    per_rule = [
        {
            "rule": name,
            "margin_dx": rules[name],
            "difference_from_primary_dx": rules[name] - rules[primary],
            "full_displacement_dx": float(family[name]["cumulative_dn_dx"]["full"]),
        }
        for name in sorted(rules)
    ]
    primary_full = float(family[primary]["cumulative_dn_dx"]["full"])
    legs = []
    for name in sorted(alternatives):
        leg = dict(family[name]["cumulative_dn_dx"])
        leg.update(
            {
                "rule": name,
                "margin_dx": rules[name],
                "delta_full_dx": leg["full"] - primary_full,
                "delta_adv_dx": leg["adv"] - family[primary]["cumulative_dn_dx"]["adv"],
                "delta_net_ch_dx": (leg["ch"] + leg["imp"])
                - (family[primary]["cumulative_dn_dx"]["ch"] + family[primary]["cumulative_dn_dx"]["imp"]),
            }
        )
        legs.append(leg)
    recorded = x2.get("cross_rule_comparison", [])
    recorded_rescaling = [
        {
            "rule": entry["rule"],
            "rescaling_to_the_primary_full_displacement": entry["rescaling_to_the_primary_full_displacement"],
            "margin_primary_units_dx": entry["margin_primary_units_dx"],
        }
        for entry in recorded
    ]
    residuals = json.loads((TWO_PHASE / "evidence" / "l1a2x" / "residual_decomposition.json").read_text())
    common = residuals["window_where_every_measurement_exists"]
    payload = {
        "stage": STAGE,
        "source": "evidence/l1a2x/interface_attribution.json and residual_decomposition.json (sealed, unedited)",
        "raw_margins_dx": rules,
        "raw_margin_spread_dx": raw_spread,
        "per_rule": per_rule,
        "contributions_of_each_leg": legs,
        "what_the_spread_is_made_of": {
            "coefficient_leg_over_dx": legs[1]["margin_dx"] - rules[primary] if len(legs) > 1 else None,
            "sampling_point_leg_over_dx": legs[0]["margin_dx"] - rules[primary] if legs else None,
            "note": (
                "the spread between the prespecified rules is a genuine, dimensional disagreement: it "
                "carries both the difference between sampling the increment at one point and over the "
                "support cell, and the difference between the frozen t = 0 coefficient and the local "
                "coefficient. It is reported here in dx, unreduced"
            ),
        },
        "how_l1a2x_reported_it": {
            "recorded_margin_spread_across_rules_dx": x2["margin_spread_across_rules_dx"],
            "recorded_rescaling": recorded_rescaling,
            "criticism": (
                "the recorded spread was the spread of the *rescaled* margins, that is each rule's margin "
                "divided by that rule's own full displacement in primary units; that normalisation is "
                "exactly one and divides a genuine difference by the effect it is supposed to measure, so "
                "the comparison band of L1A-2x (6.4e-15 dx) understated its own rule disagreement. This "
                "stage does not inherit that band and does not use any such rescaling"
            ),
            "arithmetic_note": (
                "for the two alternative rules the rescaled margin came out numerically equal to the raw "
                "margin (to 4e-17 dx) although the rescaling factors are 0.976359 and 0.997184; the "
                "recorded field is therefore not a correct quotient of the quantities it names, and it is "
                "recomputed here from the sealed margins and displacements rather than trusted"
            ),
        },
        "prespecified_comparison_error_of_this_stage": {
            "rule": (
                "comparison_error = max(raw rule spread, measured walk noise floor, arc/root agreement), "
                "all in dx and all measured in this run; the verdict needs margin > dominance factor x "
                "comparison_error"
            ),
            "dominance_factor": DOMINANCE_FACTOR,
            "walk_noise_floor_dx": WALK_NOISE_FLOOR_DX,
        },
        "l1a2x_measured_reconstruction_residuals_dx": {
            "interpolation": residuals["cumulative_abs_band"]["R_interpolation"],
            "support_migration": residuals["cumulative_abs_band"]["R_support_migration"],
            "nonlinearity": residuals["cumulative_abs_band"]["R_nonlinearity"],
            "operator_splitting": residuals["cumulative_abs_band"]["R_operator_splitting"],
            "linearisation_coefficient_drift": residuals["cumulative_abs_band"]["R_linearisation_coefficient_drift"],
            "total_band": residuals["total_residual_band_dx"],
            "tracked_edge_coverage": residuals["tracked_edge_coverage"],
            "common_window_total_band_dx": common["total_band_dx"],
            "common_window_linearisation_overshoot_dx": common["linearisation_overshoot_dx"],
        },
        "rule": (
            "the comparison error of L1A-2x is recomputed from its sealed numbers and reported in raw dx; "
            "the old stage's payloads are left exactly as they are and no new conclusion is written into "
            "them"
        ),
    }
    return payload


# ---------------------------------------------------------------------------
# Section C - phase 2: the one sanctioned window, with reconstruction-stable capture
# ---------------------------------------------------------------------------
def run_window(macro_steps: int = MAX_MACRO_STEPS) -> dict[str, Any]:
    """Run the one sanctioned window and measure the tracked branch at every frozen substep.

    The solver path is the frozen one, re-executed by the L1A-2x kernel capture; the measurement is the
    new branch readout. Exactly one case, one ``dt``, one initial state, at most ``MAX_MACRO_STEPS`` macro
    steps, and the window stops early if the frozen contact gap threshold is reached.
    """
    if macro_steps < 1 or macro_steps > MAX_MACRO_STEPS:
        raise AuditBlocked(f"macro step budget {macro_steps} is outside 1..{MAX_MACRO_STEPS}")
    bundle, state0 = stage_2x._bundle_and_state()
    p = bundle.p
    solid = bundle.solid
    pf.validate_x64_for_phase_storage(p.phase_storage_model)
    dx = float(p.dx)
    dy = float(p.dy)
    eps = float(p.eps)
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    aperture_x = np.asarray(pf.phase_face_apertures(solid, p)[0], dtype=np.float64)
    aperture_y = np.asarray(pf.phase_face_apertures(solid, p)[1], dtype=np.float64)
    support = stage_2x.fixed_support(bundle, state0)
    spec = readout.branch_definition(p, np.asarray(state0.phi, dtype=np.float64), support)
    wall_normal = tuple(float(value) for value in spec["wall_normal_into_solid"])
    pinned_point = tuple(float(value) for value in support["point"])
    pinned_cells = [tuple(int(value) for value in support["cell"])]
    pinned_cells.append((min(pinned_cells[0][0] + 1, int(p.Nx) - 1), pinned_cells[0][1]))
    gradient0_norm = float(support["gradient_norm"])
    coefficient0_over_dx = (
        -float(np.dot(np.array(spec["interface_normal"]), np.array(wall_normal))) / gradient0_norm / dx
    )
    jit_step = jax.jit(pf.step, static_argnums=(2,))
    raw_eager = state0
    raw_jitted = state0
    equivalence: dict[str, Any] = {"per_macro": []}
    phi, u, v, t = state0.phi, state0.u, state0.v, state0.t
    contact_guard: dict[str, Any] = {"triggered": False, "macro_step": None, "gap_over_dx": None}
    rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    static_rows: list[dict[str, Any]] = []
    segment_binding_switch: dict[str, Any] = {"found": False, "macro_step": None, "substep": None}
    previous_roots: dict[str, float] = {
        "adv": float(spec["t0_root_over_dx"]),
        "ch": float(spec["t0_root_over_dx"]),
        "imp": float(spec["t0_root_over_dx"]),
        "full": float(spec["t0_root_over_dx"]),
        "chpair": float(spec["t0_root_over_dx"]),
        "normal_ray": 0.0,
    }
    for macro in range(macro_steps):
        macro_base_field = np.asarray(phi, dtype=np.float64)
        macro_base_root = float(previous_roots["full"])
        macro_base_levels = [float(macro_base_field[cell[0], cell[1]]) for cell in pinned_cells]
        macro_adv: np.ndarray | None = None
        macro_ch: np.ndarray | None = None
        macro_imp: np.ndarray | None = None
        macro_full: np.ndarray | None = None
        for _substep in range(SUBSTEPS_PER_MACRO):
            base_field = np.asarray(phi, dtype=np.float64)
            base_root = previous_roots["full"]
            t_before = float(np.asarray(t))
            mass_before = stage_2x._mass(volume, phi)
            (phi, u, v, t), ingredient = stage_2x.substep_capture(phi, u, v, t, bundle)
            mass_after = stage_2x._mass(volume, phi)
            d_adv = np.asarray(ingredient["d_adv"], dtype=np.float64)
            d_ch = np.asarray(ingredient["d_ch"], dtype=np.float64)
            d_imp = np.asarray(ingredient["d_imp"], dtype=np.float64)
            d_full = np.asarray(ingredient["d_full"], dtype=np.float64)
            recombination = (d_adv + d_ch + d_imp) - d_full
            if macro_adv is None:
                macro_adv, macro_ch, macro_imp, macro_full = (
                    d_adv.copy(),
                    d_ch.copy(),
                    d_imp.copy(),
                    d_full.copy(),
                )
            else:
                macro_adv = macro_adv + d_adv
                macro_ch = macro_ch + d_ch
                macro_imp = macro_imp + d_imp
                macro_full = macro_full + d_full
            partial_fields = {
                "adv": base_field + d_adv,
                "ch": base_field + d_ch,
                "imp": base_field + d_imp,
                "chpair": base_field + d_ch + d_imp,
            }
            full_root_record = readout.branch_root(phi, spec, previous_roots["full"], with_arc=True)
            ingredient_records = {
                name: readout.branch_root(field, spec, previous_roots[name], with_arc=False)
                for name, field in partial_fields.items()
            }
            normal_ray_record = readout.branch_root(
                phi, spec, previous_roots["normal_ray"], ray="interface_normal", with_arc=False
            )
            if full_root_record["matched"] is not None:
                previous_roots["full"] = float(full_root_record["matched"]["y_over_dx"])
            for name, record in ingredient_records.items():
                if record["matched"] is not None:
                    previous_roots[name] = float(record["matched"]["y_over_dx"])
            if normal_ray_record["matched"] is not None:
                previous_roots["normal_ray"] = float(normal_ray_record["matched"]["s_over_dx"])
            segment_index = (
                None if full_root_record["matched"] is None else full_root_record["matched"]["segment_index"]
            )
            if (
                segment_index is not None
                and not segment_binding_switch["found"]
                and rows
                and rows[-1]["tracked_segment_index"] is not None
                and segment_index != rows[-1]["tracked_segment_index"]
            ):
                segment_binding_switch = {
                    "found": True,
                    "macro_step": macro + 1,
                    "substep": len(rows),
                    "t": float(np.asarray(t)),
                    "from_segment_index": rows[-1]["tracked_segment_index"],
                    "to_segment_index": segment_index,
                    "note": (
                        "the reconstructed crossing moved from one grid segment to the next; this is the "
                        "support change that stopped L1A-2x at its macro step 25, and the readout reports it "
                        "as an ordinary recorded event because the tracked root is matched by continuation"
                    ),
                }
            gradient = stage_2x._gradient_at(phi, pinned_point, bundle)
            gradient_norm = float(np.hypot(gradient[0], gradient[1]))
            gap_row = audit._local_gap(np.asarray(phi, dtype=np.float64), 0.5, solid, p)
            rows.append(
                {
                    "substep": len(rows),
                    "macro_step": macro + 1,
                    "t_before": t_before,
                    "t_after": float(np.asarray(t)),
                    "mass_before": mass_before,
                    "mass_after": mass_after,
                    "mass_adv": stage_2x._mass(volume, d_adv),
                    "mass_ch": stage_2x._mass(volume, d_ch),
                    "mass_imp": stage_2x._mass(volume, d_imp),
                    "mass_complete": stage_2x._mass(volume, d_full),
                    "mass_residual": mass_after - mass_before - stage_2x._mass(volume, d_full),
                    "telescoping_adv": float(np.sum(volume * np.asarray(ingredient["div_adv"], dtype=np.float64))),
                    "telescoping_ch": float(np.sum(volume * np.asarray(ingredient["div_ch"], dtype=np.float64))),
                    "recombination_linf": float(np.max(np.abs(recombination))),
                    "closed_face_flux_max": stage_2x._closed_face_flux_max(
                        [ingredient["adv_x"], ingredient["adv_y"], ingredient["ch_x"], ingredient["ch_y"]],
                        [aperture_x, aperture_y, aperture_x, aperture_y],
                    ),
                    "cg_iterations": int(ingredient["solve_info"].iterations),
                    "cg_relative_residual": float(ingredient["solve_info"].relative_residual),
                    "cg_converged": bool(ingredient["solve_info"].converged),
                    "dphi_adv_at_point": stage_2x._sample(ingredient["d_adv"], pinned_point, bundle),
                    "dphi_ch_at_point": stage_2x._sample(ingredient["d_ch"], pinned_point, bundle),
                    "dphi_imp_at_point": stage_2x._sample(ingredient["d_imp"], pinned_point, bundle),
                    "dphi_full_at_point": stage_2x._sample(ingredient["d_full"], pinned_point, bundle),
                    "dphi_adv_at_cells": [float(d_adv[cell[0], cell[1]]) for cell in pinned_cells],
                    "dphi_ch_at_cells": [float(d_ch[cell[0], cell[1]]) for cell in pinned_cells],
                    "dphi_imp_at_cells": [float(d_imp[cell[0], cell[1]]) for cell in pinned_cells],
                    "dphi_full_at_cells": [float(d_full[cell[0], cell[1]]) for cell in pinned_cells],
                    "base_levels_at_cells": [float(base_field[cell[0], cell[1]]) for cell in pinned_cells],
                    "full_levels_at_cells": [
                        float(np.asarray(phi, dtype=np.float64)[cell[0], cell[1]]) for cell in pinned_cells
                    ],
                    "phi_at_point": stage_2x._sample(phi, pinned_point, bundle),
                    "gradient_norm_at_point": gradient_norm,
                    "interface_normal_velocity_at_point": float(
                        stage_2x._sample(u, pinned_point, bundle) * wall_normal[0]
                        + stage_2x._sample(v, pinned_point, bundle) * wall_normal[1]
                    ),
                    "root_full_over_dx": None
                    if full_root_record["matched"] is None
                    else full_root_record["matched"]["y_over_dx"],
                    "root_full_status": full_root_record["status"],
                    "tracked_segment_index": segment_index,
                    "root_full_at_shared_node": bool(
                        full_root_record["matched"] and full_root_record["matched"]["at_shared_node"]
                    ),
                    "arc_status": None if full_root_record.get("arc") is None else full_root_record["arc"]["status"],
                    "arc_segments_traced": None
                    if full_root_record.get("arc") is None
                    else full_root_record["arc"]["segments_traced"],
                    "root_adv_over_dx": None
                    if ingredient_records["adv"]["matched"] is None
                    else ingredient_records["adv"]["matched"]["y_over_dx"],
                    "root_ch_over_dx": None
                    if ingredient_records["ch"]["matched"] is None
                    else ingredient_records["ch"]["matched"]["y_over_dx"],
                    "root_imp_over_dx": None
                    if ingredient_records["imp"]["matched"] is None
                    else ingredient_records["imp"]["matched"]["y_over_dx"],
                    "root_chpair_over_dx": None
                    if ingredient_records["chpair"]["matched"] is None
                    else ingredient_records["chpair"]["matched"]["y_over_dx"],
                    "root_ingredient_statuses": {name: record["status"] for name, record in ingredient_records.items()},
                    "root_normal_ray_over_dx": None
                    if normal_ray_record["matched"] is None
                    else normal_ray_record["matched"]["s_over_dx"],
                    "root_normal_ray_status": normal_ray_record["status"],
                    "base_root_over_dx": base_root,
                    "gap_global_over_dx": None if gap_row.get("gap") is None else float(gap_row["gap"]) / dx,
                    "gap_global_cell": None if gap_row.get("gap_cell") is None else list(gap_row["gap_cell"]),
                }
            )
        # --- static reconstruction and order decomposition at this macro boundary (no solver work) -----
        assert macro_adv is not None and macro_ch is not None and macro_imp is not None and macro_full is not None
        static_rows.append(
            _static_reconstruction_row(
                macro=macro + 1,
                t=float(np.asarray(t)),
                base_field=macro_base_field,
                full=macro_full,
                adv=macro_adv,
                ch=macro_ch,
                imp=macro_imp,
                spec=spec,
                anchor_over_dx=macro_base_root,
                resolved_root_over_dx=rows[-1]["root_full_over_dx"],
                base_levels=macro_base_levels,
                pinned_cells=pinned_cells,
            )
        )
        raw_eager, eager_info = pf.step_with_diagnostics(raw_eager, solid, p)
        raw_jitted = jax.block_until_ready(jit_step(raw_jitted, solid, p))
        phi_np = np.asarray(phi, dtype=np.float64)
        jitted_np = np.asarray(raw_jitted.phi, dtype=np.float64)
        gap_capture = rows[-1]["gap_global_over_dx"]
        gap_public = audit._local_gap(jitted_np, 0.5, solid, p)
        gap_public_dx = float(gap_public["gap"]) / dx if gap_public.get("gap") is not None else None
        uniqueness = stage_2x._contour_uniqueness(phi, bundle)
        equivalence["per_macro"].append(
            {
                "macro_step": macro + 1,
                "max_phi_diff_eager": float(np.max(np.abs(phi_np - np.asarray(raw_eager.phi, dtype=np.float64)))),
                "max_phi_diff_jitted": float(np.max(np.abs(phi_np - jitted_np))),
                "max_velocity_diff_eager": float(
                    max(
                        np.max(np.abs(np.asarray(u) - np.asarray(raw_eager.u))),
                        np.max(np.abs(np.asarray(v) - np.asarray(raw_eager.v))),
                    )
                ),
                "max_velocity_diff_jitted": float(
                    max(
                        np.max(np.abs(np.asarray(u) - np.asarray(raw_jitted.u))),
                        np.max(np.abs(np.asarray(v) - np.asarray(raw_jitted.v))),
                    )
                ),
                "cg_iterations_eager": [int(value) for value in np.asarray(eager_info.implicit_iterations)],
                "cg_converged_eager": bool(np.all(np.asarray(eager_info.implicit_converged))),
                "gap_capture_over_dx": gap_capture,
                "gap_public_step_over_dx": gap_public_dx,
                "gap_difference_over_dx": (
                    abs(float(gap_capture) - float(gap_public_dx))
                    if gap_capture is not None and gap_public_dx is not None
                    else None
                ),
                "contour_uniqueness": uniqueness,
            }
        )
        macro_rows.append(
            {
                "macro_step": macro + 1,
                "t": rows[-1]["t_after"],
                "root_full_over_dx": rows[-1]["root_full_over_dx"],
                "root_full_status": rows[-1]["root_full_status"],
                "tracked_segment_index": rows[-1]["tracked_segment_index"],
                "root_normal_ray_over_dx": rows[-1]["root_normal_ray_over_dx"],
                "gap_global_over_dx": rows[-1]["gap_global_over_dx"],
                "gap_global_cell": rows[-1]["gap_global_cell"],
                "gap_public_step_over_dx": gap_public_dx,
                "gap_difference_over_dx": equivalence["per_macro"][-1]["gap_difference_over_dx"],
                "phi_at_point": rows[-1]["phi_at_point"],
                "gradient_norm_at_point": rows[-1]["gradient_norm_at_point"],
                "interface_normal_velocity_at_point": rows[-1]["interface_normal_velocity_at_point"],
                "contour_uniqueness": uniqueness,
                "arc_status": rows[-1]["arc_status"],
            }
        )
        if gap_capture is not None and float(gap_capture) <= CONTACT_GAP_THRESHOLD_DX:
            contact_guard.update({"triggered": True, "macro_step": macro + 1, "gap_over_dx": float(gap_capture)})
            break
    parameters = {
        "dx": dx,
        "dy": dy,
        "eps": eps,
        "dt": float(p.dt),
        "substep_dt": float(p.dt) / SUBSTEPS_PER_MACRO,
        "N": PROFILE_N,
        "u_impact": float(bundle.u_impact),
        "eps_over_dx": eps / dx,
        "M": float(p.M),
        "ch_solver_rtol": float(p.ch_solver_rtol),
        "ch_solver_max_iterations": int(p.ch_solver_max_iterations),
    }
    return {
        "case_name": CASE_NAME,
        "profile_N": PROFILE_N,
        "requested_dt": REQUESTED_DT,
        "macro_steps_requested": macro_steps,
        "macro_steps_run": len(macro_rows),
        "substeps_per_macro": SUBSTEPS_PER_MACRO,
        "substeps_recorded": len(rows),
        "t_final": macro_rows[-1]["t"] if macro_rows else 0.0,
        "contact_guard": contact_guard,
        "contact_threshold_over_dx": CONTACT_GAP_THRESHOLD_DX,
        "support": support,
        "pinned_cells": [list(cell) for cell in pinned_cells],
        "pinned_branch": {
            key: value for key, value in readout.spec_payload(spec).items() if key not in {"rules", "constants"}
        }
        | {"rules": dict(readout.PRESPECIFIED_RULES), "constants": dict(spec["constants"])},
        "parameters": parameters,
        "linearisation_reference": {
            "coefficient0_over_dx": coefficient0_over_dx,
            "coefficient0_length_units": -float(np.dot(np.array(spec["interface_normal"]), np.array(wall_normal)))
            / gradient0_norm,
            "gradient_norm_at_t0": gradient0_norm,
            "interface_normal": list(spec["interface_normal"]),
            "wall_normal_into_solid": list(wall_normal),
            "rule": (
                "the L1A-2x linearisation coefficient rebuilt from the same frozen t = 0 normal, gradient and "
                "wall normal, so the linearised rules of this stage are the arithmetic analogue of the "
                "previous stage's rules"
            ),
        },
        "initial_state": {
            "t": float(np.asarray(state0.t)),
            "phi_hash": _array_sha(state0.phi),
            "u_hash": _array_sha(state0.u),
            "v_hash": _array_sha(state0.v),
        },
        "final_state": {
            "t": float(np.asarray(t)),
            "phi_hash": _array_sha(phi),
            "u_hash": _array_sha(u),
            "v_hash": _array_sha(v),
        },
        "rows": rows,
        "macro_rows": macro_rows,
        "static_reconstruction_rows": static_rows,
        "segment_binding_switch": segment_binding_switch,
        "capture_equivalence": equivalence,
        "prohibitions": {
            "control_arm_runs": 0,
            "refinement_runs": 0,
            "contact_events_scored": 0,
            "t8_recomputed": False,
            "l1b_data_generated": False,
            "ml_training_runs": 0,
            "solver_source_edited": False,
            "initializer_promoted": False,
        },
        "equivalence_note": (
            "the eager and jitted public steps are advanced from the same initial state in lockstep with the "
            "capture. They duplicate the single sanctioned trajectory and exist only to verify that the "
            "capture reproduces the frozen public step; no physical conclusion is drawn from them and no "
            "control arm is run."
        ),
    }


def _wall_normal_displacement(root_from: float, root_to: float, wall_normal: tuple[float, float]) -> float:
    return readout.displacement_over_dx_from_heights(root_from, root_to, wall_normal)


def _static_reconstruction_row(
    *,
    macro: int,
    t: float,
    base_field: np.ndarray,
    full: np.ndarray,
    adv: np.ndarray,
    ch: np.ndarray,
    imp: np.ndarray,
    spec: dict[str, Any],
    anchor_over_dx: float,
    resolved_root_over_dx: float | None,
    base_levels: list[float],
    pinned_cells: list[tuple[int, int]],
) -> dict[str, Any]:
    """The static reconstruction and the symmetric order decomposition of one macro step.

    Everything here is a pure function of what the frozen solver already produced for that macro step: its
    base field, the three ingredient increments and the complete increment, recombined statically. No extra
    solver work is performed and no trajectory is advanced.
    """

    def root(field: np.ndarray) -> tuple[float | None, str]:
        record = readout.branch_root(field, spec, anchor_over_dx, with_arc=False)
        matched = record["matched"]
        return (None if matched is None else float(matched["y_over_dx"])), str(record["status"])

    y_base, status_base = root(base_field)
    y_full, status_full = root(base_field + full)
    y_sum, status_sum = root(base_field + (adv + ch + imp))
    y_half, status_half = root(base_field + 0.5 * full)
    y_pair, status_pair = root(base_field + (ch + imp))
    components = {}
    statuses = {"base": status_base, "full": status_full, "ingredient_sum": status_sum, "half": status_half}
    for name, increment in (("adv", adv), ("ch", ch), ("imp", imp)):
        components[name], statuses[name] = root(base_field + increment)
    leave_one_out = {}
    for name, increment in (("adv", adv), ("ch", ch), ("imp", imp)):
        leave_one_out[name], statuses[f"without_{name}"] = root(base_field + (full - increment))
    forward_terms = [value for value in components.values() if value is not None]
    backward_terms = [value for value in leave_one_out.values() if value is not None]
    static_forward = None
    if y_base is not None and len(forward_terms) == 3:
        static_forward = float(y_base) + float(np.sum([value - float(y_base) for value in forward_terms]))
    static_backward = None
    if y_full is not None and len(backward_terms) == 3:
        static_backward = float(y_full) - float(np.sum([value - float(y_full) for value in backward_terms]))
    static_ch_pair = None
    if y_base is not None and components.get("ch") is not None and components.get("imp") is not None:
        static_ch_pair = (
            float(y_base) + (float(components["ch"]) - float(y_base)) + (float(components["imp"]) - float(y_base))
        )
    increment_at_cells = [float(full[cell[0], cell[1]]) for cell in pinned_cells]
    fractions = [
        None if delta == 0.0 else (0.5 - float(level)) / float(delta)
        for level, delta in zip(base_levels, increment_at_cells, strict=True)
    ]
    row: dict[str, Any] = {
        "macro_step": macro,
        "t": t,
        "anchor_over_dx": anchor_over_dx,
        "root_base_over_dx": y_base,
        "root_full_over_dx": y_full,
        "root_ingredient_sum_over_dx": y_sum,
        "root_half_increment_over_dx": y_half,
        "root_ch_pair_over_dx": y_pair,
        "root_component_over_dx": components,
        "root_leave_one_out_over_dx": leave_one_out,
        "static_reconstruction_forward_over_dx": static_forward,
        "static_reconstruction_backward_over_dx": static_backward,
        "static_reconstruction_ch_pair_over_dx": static_ch_pair,
        "resolved_root_over_dx": resolved_root_over_dx,
        "statuses": statuses,
        "base_levels_at_cells": list(base_levels),
        "increment_at_cells": increment_at_cells,
        "level_based_fraction": fractions,
        "resolution": (
            "the reconstruction is static: the base field of the macro step, the three ingredient increments "
            "and the complete increment are recombined algebraically and read with the same branch readout, so "
            "the deviation from the directly measured field isolates the nonlinearity of the contour mapping"
        ),
    }
    if y_full is not None and y_sum is not None:
        row["R_operator_recombination_over_dx"] = abs(float(y_sum) - float(y_full))
    if y_full is not None and static_forward is not None:
        row["R_static_forward_over_dx"] = abs(static_forward - float(y_full))
    if y_full is not None and static_backward is not None:
        row["R_static_backward_over_dx"] = abs(static_backward - float(y_full))
    if static_forward is not None and static_backward is not None:
        row["R_order_symmetry_over_dx"] = abs(static_forward - static_backward)
    if y_base is not None and y_half is not None and y_full is not None:
        row["R_half_increment_nonlinearity_over_dx"] = abs(float(y_full) + float(y_base) - 2.0 * float(y_half))
    if static_ch_pair is not None and y_pair is not None:
        row["R_static_ch_pair_over_dx"] = abs(static_ch_pair - float(y_pair))
    return row


def interface_attribution(ledger: dict[str, Any]) -> dict[str, Any]:
    """The attribution table: ingredient displacements, rule margins and the raw comparison error."""
    rows = ledger["rows"]
    wall_normal = tuple(ledger["support"]["wall_normal_into_solid"])
    coefficient0_over_dx = float(ledger["linearisation_reference"]["coefficient0_over_dx"])
    ingredients = ("adv", "ch", "imp")
    cumulative_series: dict[str, list[float]] = {name: [] for name in ingredients}
    own_base_series: dict[str, list[float]] = {name: [] for name in ingredients}
    full_series: list[float] = []
    ch_pair_direct = 0.0
    previous_cumulative = {name: float(ledger["pinned_branch"]["t0_root_over_dx"]) for name in ingredients}
    for row in rows:
        base_root = float(row["base_root_over_dx"])
        for name in ingredients:
            root = row[f"root_{name}_over_dx"]
            if root is None:
                cumulative_series[name].append(0.0)
                own_base_series[name].append(0.0)
                continue
            cumulative_series[name].append(
                _wall_normal_displacement(previous_cumulative[name], float(root), wall_normal)
            )
            own_base_series[name].append(_wall_normal_displacement(base_root, float(root), wall_normal))
            previous_cumulative[name] = float(root)
        if row["root_full_over_dx"] is not None:
            full_series.append(_wall_normal_displacement(base_root, float(row["root_full_over_dx"]), wall_normal))
        if row["root_chpair_over_dx"] is not None:
            ch_pair_direct += _wall_normal_displacement(base_root, float(row["root_chpair_over_dx"]), wall_normal)
    cumulative = {name: float(np.sum(values)) for name, values in cumulative_series.items()}
    cumulative["ch_pair"] = cumulative["ch"] + cumulative["imp"]
    own_base = {name: float(np.sum(values)) for name, values in own_base_series.items()}
    own_base["ch_pair"] = own_base["ch"] + own_base["imp"]
    total_displacement = float(np.sum(full_series))
    walk_noise = float(
        max(
            abs(own_base_series[name][index] - cumulative_series[name][index])
            for name in ingredients
            for index in range(len(rows))
        )
    )
    linearised = {
        name: float(np.sum([coefficient0_over_dx * row[f"dphi_{name}_at_point"] for row in rows]))
        for name in ("adv", "ch", "imp", "full")
    }
    linearised["ch_pair"] = linearised["ch"] + linearised["imp"]
    linearised_cell = {
        name: float(np.sum([coefficient0_over_dx * row[f"dphi_{name}_at_cells"][0] for row in rows]))
        for name in ("adv", "ch", "imp", "full")
    }
    linearised_cell["ch_pair"] = linearised_cell["ch"] + linearised_cell["imp"]
    normal_ray_total = float(
        np.sum([-float(row["root_normal_ray_over_dx"]) for row in rows if row["root_normal_ray_over_dx"] is not None])
    )

    def _margin(advective: float, chemical_pair: float) -> float:
        return abs(advective) - abs(chemical_pair)

    rule_margins = {
        "RULE_A_GEOMETRIC_TRACKED_BRANCH": _margin(cumulative["adv"], cumulative["ch_pair"]),
        "RULE_B_GEOMETRIC_OWN_BASE": _margin(own_base["adv"], own_base["ch_pair"]),
        "RULE_D_LINEARISED_POINT_T0_COEFFICIENT": _margin(linearised["adv"], linearised["ch_pair"]),
        "RULE_E_LINEARISED_SUPPORT_CELL_T0_COEFFICIENT": _margin(linearised_cell["adv"], linearised_cell["ch_pair"]),
    }
    primary = "RULE_A_GEOMETRIC_TRACKED_BRANCH"
    comparison_terms = {name: rule_margins[name] - rule_margins[primary] for name in rule_margins if name != primary}
    rule_spread = max((abs(value) for value in comparison_terms.values()), default=0.0)
    per_macro = []
    for macro_row in ledger["macro_rows"]:
        index = int(macro_row["macro_step"])
        positions = [position for position, row in enumerate(rows) if int(row["macro_step"]) == index]
        per_macro.append(
            {
                "macro_step": index,
                "t": macro_row["t"],
                "dn_adv": float(np.sum([cumulative_series["adv"][position] for position in positions])),
                "dn_ch_explicit": float(np.sum([cumulative_series["ch"][position] for position in positions])),
                "dn_ch_implicit": float(np.sum([cumulative_series["imp"][position] for position in positions])),
                "dn_full": float(np.sum([full_series[position] for position in positions])),
                "root_full_over_dx": macro_row["root_full_over_dx"],
                "tracked_segment_index": macro_row["tracked_segment_index"],
                "root_status": macro_row["root_full_status"],
            }
        )
        per_macro[-1]["dn_ch_pair"] = per_macro[-1]["dn_ch_explicit"] + per_macro[-1]["dn_ch_implicit"]
    ch_pair_magnitude = abs(cumulative["ch"]) + abs(cumulative["imp"])
    return {
        "case_name": ledger["case_name"],
        "primary_rule": primary,
        "rules": list(rule_margins),
        "cumulative_dn_dx": {
            "adv": cumulative["adv"],
            "ch_explicit": cumulative["ch"],
            "ch_implicit": cumulative["imp"],
            "ch_pair": cumulative["ch_pair"],
            "ch_pair_direct_over_dx": ch_pair_direct,
            "full": total_displacement,
            "normal_ray_full_dx": normal_ray_total,
        },
        "own_base_dn_dx": own_base,
        "linearised_dn_dx": linearised,
        "linearised_cell_dn_dx": linearised_cell,
        "rule_margins_dx": rule_margins,
        "comparison_terms_dx": comparison_terms,
        "raw_rule_spread_dx": rule_spread,
        "walk_noise_floor_dx": walk_noise,
        "cahn_hilliard_pairing": {
            "reported_as": "one Cahn-Hilliard update whose two halves are the explicit flux and the implicit solve",
            "explicit_dx": cumulative["ch"],
            "implicit_dx": cumulative["imp"],
            "pair_dx": cumulative["ch_pair"],
            "pair_directly_measured_dx": ch_pair_direct,
            "cancellation_ratio": (abs(cumulative["ch_pair"]) / ch_pair_magnitude) if ch_pair_magnitude > 0 else None,
            "forbidden_claim": (
                "the two halves are never presented as two independent physical mechanisms and no verdict is "
                "attached to either half separately"
            ),
        },
        "per_macro": per_macro,
        "per_substep": {
            "cumulative": {name: [float(value) for value in values] for name, values in cumulative_series.items()},
            "own_base": {name: [float(value) for value in values] for name, values in own_base_series.items()},
            "full": [float(value) for value in full_series],
        },
    }


def static_reconstruction_study(ledger: dict[str, Any]) -> dict[str, Any]:
    """The static component reconstruction and the symmetric order decomposition of the contour mapping.

    For every macro step the reconstruction is *static*: the base field, the three ingredient increments and
    the complete increment are recombined algebraically and read with the same branch readout. Three things
    are measured, all in raw ``dx``: the algebraic recombination of the ingredients against the complete
    increment, the forward and backward one-shot reconstructions of the crossing against the directly
    measured crossing (and their symmetry), and the symmetric half-increment nonlinearity.
    """
    rows = ledger["static_reconstruction_rows"]

    def collect(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    forward = collect("R_static_forward_over_dx")
    backward = collect("R_static_backward_over_dx")
    symmetry = collect("R_order_symmetry_over_dx")
    recombination = collect("R_operator_recombination_over_dx")
    half_increment = collect("R_half_increment_nonlinearity_over_dx")
    ch_pair = collect("R_static_ch_pair_over_dx")
    fractions = [
        abs(float(value)) for row in rows for value in (row.get("level_based_fraction") or []) if value is not None
    ]
    inside = [
        1
        for row in rows
        for value in (row.get("level_based_fraction") or [])
        if value is not None and 0.0 <= float(value) <= 1.0
    ]
    statuses = sorted({str(value) for row in rows for value in row["statuses"].values()})
    resolutions = [float(row["anchor_over_dx"]) for row in rows]
    starts = [None if row["root_base_over_dx"] is None else float(row["root_base_over_dx"]) for row in rows]
    holds = bool(
        forward
        and backward
        and recombination
        and max(forward) <= STATIC_RECONSTRUCTION_TOLERANCE_DX
        and max(backward) <= STATIC_RECONSTRUCTION_TOLERANCE_DX
        and max(recombination) <= STATIC_RECONSTRUCTION_TOLERANCE_DX
    )
    return {
        "stage": STAGE,
        "method": (
            "for every macro step the base field of the step, the three ingredient increments of its three "
            "frozen substeps and the complete increment are recombined statically and read with the same "
            "branch readout pinned on the same wall-normal ray. Nothing is scaled by a level fraction: the "
            "complete increment is applied as it is produced, and the order decomposition is taken in both "
            "directions - once from the base crossing and once by removing each ingredient from the complete "
            "crossing - so the nonlinearity of the contour mapping is measured symmetrically"
        ),
        "rows": rows,
        "macros_total": len(rows),
        "macros_with_a_resolved_support_cell": len(forward),
        "max_static_reconstruction_forward_over_dx": max(forward) if forward else None,
        "max_static_reconstruction_backward_over_dx": max(backward) if backward else None,
        "max_order_symmetry_over_dx": max(symmetry) if symmetry else None,
        "max_operator_recombination_over_dx": max(recombination) if recombination else None,
        "max_half_increment_nonlinearity_over_dx": max(half_increment) if half_increment else None,
        "max_static_ch_pair_reconstruction_over_dx": max(ch_pair) if ch_pair else None,
        "max_static_forward_over_dx": max(forward) if forward else None,
        "max_static_backward_over_dx": max(backward) if backward else None,
        "max_static_ch_pair_over_dx": max(ch_pair) if ch_pair else None,
        "max_frozen_sigma_deviation_over_dx": max(forward) if forward else None,
        "max_frozen_sigma_chpair_deviation_over_dx": max(ch_pair) if ch_pair else None,
        "tolerance_over_dx": STATIC_RECONSTRUCTION_TOLERANCE_DX,
        "readout_statuses_seen": statuses,
        "anchors_over_dx": resolutions,
        "tracked_roots_at_the_macro_bases_over_dx": starts,
        "level_based_fraction_max_abs": max(fractions) if fractions else None,
        "level_based_fraction_inside_the_unit_interval": len(inside),
        "level_based_fraction_total": len(fractions),
        "level_based_fraction_note": (
            "the level-based fraction (0.5 - cell level) / cell increment is recorded only as a diagnostic of "
            "why it is not a usable interpolation weight: it resolves the *cell* level onto the threshold "
            "rather than the *crossing*, so whenever the tracked crossing is not sweeping through that cell "
            "centre inside the step the fraction leaves the unit interval and the rescaled field is unrelated "
            "to the trajectory; the reconstruction above therefore never rescales the increment"
        ),
        "support_cell_resolved": bool(forward),
        "holds": holds,
        "nonlinearity_attribution": (
            "the total displacement of a macro step is attributed twice: forward, as the sum of the "
            "displacements that each ingredient produces on its own on that base field, and backward, as the "
            "displacement the complete update loses when an ingredient is removed. The difference between the "
            "two is the order-symmetry term, the difference between either sum and the directly measured "
            "crossing is the nonlinearity of the contour mapping, and the difference between the recombined "
            "ingredients and the complete increment is the operator-recombination term"
        ),
    }


def residual_decomposition(ledger: dict[str, Any], attribution: dict[str, Any]) -> dict[str, Any]:
    """Every displacement residual of this stage, in raw dx, as a disagreement of two measurements.

    The decorated :func:`post_window_diagnosis` records what the failed gates of this stage turned out to be
    made of. It is marked as exploratory, it promotes nothing and it changes no prespecified constant.
    """
    rows = ledger["rows"]
    cumulative = attribution["cumulative_dn_dx"]
    ingredient_sum = float(cumulative["adv"] + cumulative["ch_pair"])
    total_measured = float(cumulative["full"])
    study = ledger["static_reconstruction_rows"]

    def collect(key: str) -> list[float]:
        return [float(row[key]) for row in study if row.get(key) is not None]

    static_forward = collect("R_static_forward_over_dx")
    static_ch_pair = collect("R_static_ch_pair_over_dx")
    order_symmetry = collect("R_order_symmetry_over_dx")
    recombination = collect("R_operator_recombination_over_dx")
    half_increment = collect("R_half_increment_nonlinearity_over_dx")
    coefficient = float(ledger["linearisation_reference"]["coefficient0_over_dx"])
    operator_split = float(
        np.max(
            [
                abs(
                    stage_2x._dn(coefficient, row["dphi_full_at_point"])
                    - stage_2x._dn(
                        coefficient,
                        row["dphi_adv_at_point"] + row["dphi_ch_at_point"] + row["dphi_imp_at_point"],
                    )
                )
                for row in rows
            ]
        )
    )
    cross_ray = float(cumulative["normal_ray_full_dx"])
    own_base = attribution["own_base_dn_dx"]
    own_base_closure = float(own_base["adv"] + own_base["ch_pair"]) - total_measured
    own_base_pair_closure = float(own_base["ch_pair"]) - float(cumulative["ch_pair_direct_over_dx"])
    return {
        "terms_dx": {
            "R_attribution_closure": total_measured - ingredient_sum,
            "R_attribution_closure_own_base": own_base_closure,
            "R_ch_pair_closure": float(cumulative["ch_pair"]) - float(cumulative["ch_pair_direct_over_dx"]),
            "R_ch_pair_closure_own_base": own_base_pair_closure,
            "R_nonlinearity_point_linearisation": total_measured - float(attribution["linearised_dn_dx"]["full"]),
            "R_nonlinearity_support_cell_linearisation": total_measured
            - float(attribution["linearised_cell_dn_dx"]["full"]),
            "R_rule_spread": float(attribution["raw_rule_spread_dx"]),
            "R_walk_noise": float(attribution["walk_noise_floor_dx"]),
            "R_operator_recombination": max(recombination) if recombination else None,
            "R_static_reconstruction_forward": max(static_forward) if static_forward else None,
            "R_static_reconstruction_ch_pair": max(static_ch_pair) if static_ch_pair else None,
            "R_order_symmetry": max(order_symmetry) if order_symmetry else None,
            "R_half_increment_nonlinearity": max(half_increment) if half_increment else None,
            "R_cross_ray_full_displacement": total_measured - cross_ray,
            "R_operator_splitting": operator_split,
        },
        "measured_total_displacement_over_dx": total_measured,
        "linearised_total_displacement_over_dx": float(attribution["linearised_dn_dx"]["full"]),
        "linearised_support_cell_total_displacement_over_dx": float(attribution["linearised_cell_dn_dx"]["full"]),
        "cross_ray_total_displacement_over_dx": cross_ray,
        "definition": (
            "R_attribution_closure compares the complete update's tracked displacement with the sum of its "
            "three ingredient displacements measured the same way (the prespecified primary rule); "
            "R_attribution_closure_own_base and R_ch_pair_closure_own_base repeat that comparison for the "
            "per-substep measurement on the field each increment produces; R_ch_pair_closure compares the sum "
            "of the two Cahn-Hilliard halves with the displacement of their combined field; the two "
            "nonlinearity terms compare the geometric measurement with the arithmetic linearisations at the "
            "pinned point and over the support cell; R_rule_spread and R_walk_noise are the measured "
            "comparison-error terms; R_operator_recombination, R_static_reconstruction_forward, "
            "R_static_reconstruction_ch_pair, R_order_symmetry and R_half_increment_nonlinearity are the "
            "static reconstruction and order-decomposition terms of the macro steps; R_cross_ray_full_displacement "
            "is the independent interface-normal ray measurement of the same total; R_operator_splitting is the "
            "algebraic recombination of the ingredients through the coefficient. Every term is in dx and none "
            "is normalised"
        ),
        "post_window_diagnosis": post_window_diagnosis(ledger, attribution),
    }


def post_window_diagnosis(ledger: dict[str, Any], attribution: dict[str, Any]) -> dict[str, Any]:
    """What the withheld verdict of this stage is made of, recorded after the window without promotion rights.

    The prespecified primary rule (the cumulative walk of each ingredient's own partial field) fails its
    closure gate. The numbers show why, and they show which measurements of the same window are internally
    coherent. Nothing here is promoted: no constant, rule or gate of this stage is changed, and the verdict
    of this stage is withheld exactly because of the failure this diagnosis explains.
    """
    cumulative = attribution["cumulative_dn_dx"]
    own_base = attribution["own_base_dn_dx"]
    linearised = attribution["linearised_dn_dx"]
    linearised_cell = attribution["linearised_cell_dn_dx"]
    sealed = json.loads((TWO_PHASE / "evidence" / "l1a2x" / "interface_attribution.json").read_text())
    sealed_family = sealed["family"]
    sealed_primary = sealed_family["PRIMARY_POINT_T0_COEFFICIENT"]["cumulative_dn_dx"]
    sealed_cell = sealed_family["SUPPORT_CELL_T0_COEFFICIENT"]["cumulative_dn_dx"]
    rules = attribution["rule_margins_dx"]
    cumulative_pair_half = 0.5 * float(cumulative["ch_pair"])
    diagnosis = {
        "status": "EXPLORATORY_DIAGNOSIS_NO_PROMOTION_RIGHTS",
        "primary_rule": attribution["primary_rule"],
        "what_failed": (
            "two independent things withhold the verdict of this stage. First, the prespecified primary rule "
            "fails its attribution closure by a wide margin. Second, the frozen global contour match at the "
            "tracked support point is non-unique on every macro step, which by itself is a non-unique root "
            "and therefore an UNRESOLVED verdict. Every other guard of this stage passes - branch "
            "continuity, ingredient continuity, the step bound, the gradient floor, the probe band, the arc, "
            "the ray-local ambiguity test, the static reconstruction, conservation and capture equivalence - "
            "and the comparison error is reported in full"
        ),
        "why_the_primary_rule_fails": (
            "the primary rule walks one continuous chain per ingredient over the partial fields of the "
            "substeps. Such a chain telescopes: its total is the displacement from the t = 0 crossing to the "
            "crossing of the last field of that chain, and the last field of a chain is the physical field of "
            "a substep with that ingredient's own increment applied to it - it already contains the rest of "
            "the update, so the chain measures the complete motion of that substep, lagged, rather than the "
            "contribution of its own ingredient. The measured numbers show exactly that: the advective leg "
            "lands at the complete displacement of the window and the Cahn-Hilliard pair leg at about twice "
            "it, so the sum of the legs overshoots the complete displacement by the closure residual, "
            "irrespective of which physical term really carries the motion"
        ),
        "numbers": {
            "primary_rule_legs_dx": {
                "adv": float(cumulative["adv"]),
                "ch_pair": float(cumulative["ch_pair"]),
                "ch_explicit": float(cumulative["ch_explicit"]),
                "ch_implicit": float(cumulative["ch_implicit"]),
                "half_of_the_pair_dx": cumulative_pair_half,
                "complete_update": float(cumulative["full"]),
            },
            "own_base_legs_dx": {
                "adv": float(own_base["adv"]),
                "ch_pair": float(own_base["ch_pair"]),
                "complete_update": float(cumulative["full"]),
            },
            "point_linearised_legs_dx": {
                "adv": float(linearised["adv"]),
                "ch_pair": float(linearised["ch_pair"]),
                "complete_update": float(linearised["full"]),
            },
            "support_cell_linearised_legs_dx": {
                "adv": float(linearised_cell["adv"]),
                "ch_pair": float(linearised_cell["ch_pair"]),
                "complete_update": float(linearised_cell["full"]),
            },
            "rule_margins_dx": {name: float(value) for name, value in rules.items()},
            "sealed_l1a2x_primary_rule_dx": {name: float(value) for name, value in sealed_primary.items()},
            "sealed_l1a2x_support_cell_rule_dx": {name: float(value) for name, value in sealed_cell.items()},
        },
        "measured_structure_of_the_failure": {
            "primary_rule_closure_over_dx": abs(
                float(attribution["cumulative_dn_dx"]["full"])
                - float(attribution["cumulative_dn_dx"]["adv"])
                - float(attribution["cumulative_dn_dx"]["ch_pair"])
            ),
            "own_base_rule_closure_over_dx": abs(
                float(attribution["cumulative_dn_dx"]["full"])
                - float(attribution["own_base_dn_dx"]["adv"])
                - float(attribution["own_base_dn_dx"]["ch_pair"])
            ),
            "closure_band_over_dx": ATTRIBUTION_CLOSURE_BAND_DX,
            "note": (
                "both constructions use the same three ingredient increment fields and the same tracked "
                "branch; only the field each increment is measured against differs. The cumulative-walk "
                "construction of the prespecified primary rule does not close, and the own-base construction "
                "- evaluated against the base field of the substep that produced each increment, which is the "
                "semantics the readout design wrote down before the window ran - closes inside the "
                "prespecified band. This is recorded as an exploratory observation with no promotion "
                "rights: it promotes nothing and it changes no rule, band or constant of this stage, whose "
                "verdict stays with its failing primary rule"
            ),
            "comparison_error_note": (
                "the comparison error of the prespecified rules is dominated by the difference between the "
                "cumulative-walk construction and the arithmetic constructions (0.1238 dx), not by the "
                "nonlinearity of the contour mapping (0.0018 dx on the pinned point, 0.0016 dx over the "
                "support cell) or by the operator decomposition (2.5e-16 dx), so the primary margin of this "
                "window is smaller than its own comparison error and no dominance statement is available from "
                "these rules"
            ),
        },
        "coherent_families": {
            "own_base": (
                "the per-substep displacement of the field each increment produces on its own base field: "
                "the semantics written into the prespecified readout design before the window ran"
            ),
            "point_linearised": (
                "the arithmetic rule of L1A-2x itself: the same increments through the frozen t = 0 coefficient"
            ),
            "support_cell_linearised": "the same arithmetic rule sampled over the pinned support cell",
            "agreement": (
                "the three coherent families agree with each other and reproduce the sealed L1A-2x arithmetic "
                "numbers of the identical window; their margins put the advective leg above the Cahn-Hilliard "
                "pair by roughly the same amount, whereas the failing primary rule puts them within its own "
                "closure error of each other and at half the total"
            ),
        },
        "reading_caveats": {
            "fixed_ray_vs_tracked_branch": (
                "the independent interface-normal ray is anchored at the t = 0 crossing and therefore measures "
                "the contour sweeping past a fixed point, not the motion of the tracked branch; the two differ "
                "by the residual recorded in R_cross_ray_full_displacement, so every reading here is a "
                "statement about the tracked branch's motion along the wall normal only, never about the "
                "interface as a whole"
            ),
            "window_only": (
                "the window ends at t = 0.08 with the frozen gap still far above the event threshold, so "
                "nothing here is extrapolated toward contact"
            ),
        },
        "consequence": (
            "no mechanism is claimed in this stage: the prespecified primary rule is not a decomposition, so "
            "the prespecified verdict rule cannot be applied to it, and the tracked support point is not "
            "uniquely attached to the frozen global contour either, so the verdict stays UNRESOLVED. The "
            "coherent families are recorded here as an exploratory reading with no promotion rights, and the "
            "next action prespecifies a corrected primary rule and a unique support-point measurement before "
            "any further window is run"
        ),
    }
    return diagnosis


def conservation_and_convergence(ledger: dict[str, Any]) -> dict[str, Any]:
    """Mass, impermeable-wall, recombination and implicit-solve checks for every substep."""
    rows = ledger["rows"]
    return {
        "mass": {
            "mass_before_first": rows[0]["mass_before"],
            "mass_after_last": rows[-1]["mass_after"],
            "mass_drift_absolute": rows[-1]["mass_after"] - rows[0]["mass_before"],
            "mass_drift_relative": (rows[-1]["mass_after"] - rows[0]["mass_before"]) / rows[0]["mass_before"],
            "max_abs_substep_drift": float(np.max(np.abs([row["mass_residual"] for row in rows]))),
            "max_abs_telescoping_adv": float(np.max(np.abs([row["telescoping_adv"] for row in rows]))),
            "max_abs_telescoping_ch": float(np.max(np.abs([row["telescoping_ch"] for row in rows]))),
        },
        "wall": {
            "max_abs_flux_on_closed_faces": float(np.max([row["closed_face_flux_max"] for row in rows])),
            "rule": (
                "every face whose frozen aperture is exactly zero carries exactly zero advective and zero "
                "chemical-potential flux; no separate wall flux exists"
            ),
        },
        "implicit_solve": {
            "substeps": len(rows),
            "min_iterations": int(np.min([row["cg_iterations"] for row in rows])),
            "max_iterations": int(np.max([row["cg_iterations"] for row in rows])),
            "max_relative_residual": float(np.max([row["cg_relative_residual"] for row in rows])),
            "all_converged": bool(all(row["cg_converged"] for row in rows)),
            "rtol": ledger["parameters"]["ch_solver_rtol"],
            "iteration_cap": ledger["parameters"]["ch_solver_max_iterations"],
        },
        "recombination": {
            "max_recombination_linf": float(np.max([row["recombination_linf"] for row in rows])),
            "rule": "d_adv + d_ch + d_imp - d_full recombined cell by cell for every substep",
        },
    }


def capture_equivalence_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    """The capture-versus-public-step duplicate over the whole window."""
    per_macro = ledger["capture_equivalence"]["per_macro"]
    gap_differences = [row["gap_difference_over_dx"] for row in per_macro if row["gap_difference_over_dx"] is not None]
    return {
        "macro_steps_compared": len(per_macro),
        "max_phi_difference_vs_eager_public_step": float(np.max([row["max_phi_diff_eager"] for row in per_macro])),
        "max_phi_difference_vs_jitted_public_step": float(np.max([row["max_phi_diff_jitted"] for row in per_macro])),
        "max_velocity_difference_vs_eager_public_step": float(
            np.max([row["max_velocity_diff_eager"] for row in per_macro])
        ),
        "max_velocity_difference_vs_jitted_public_step": float(
            np.max([row["max_velocity_diff_jitted"] for row in per_macro])
        ),
        "max_gap_difference_over_dx": float(np.max(gap_differences)) if gap_differences else None,
        "thresholds": {
            "max_phi_difference": EQUIVALENCE_MAX_PHI_DIFF,
            "max_gap_difference_over_dx": EQUIVALENCE_MAX_GAP_DIFF_DX,
        },
        "meaning": (
            "the frozen public step advances its substeps inside a jitted scan, so XLA fusion changes the "
            "last-bit rounding of the elementwise update; the measured difference is reported and compared "
            "against thresholds that are orders below the displacement this stage reports"
        ),
    }


def validity_screen(
    ledger: dict[str, Any],
    attribution: dict[str, Any],
    residuals: dict[str, Any],
    study: dict[str, Any],
    equivalence: dict[str, Any],
) -> dict[str, Any]:
    """Prespecified guards; a failed guard can only withhold a verdict, never grant one."""
    rows = ledger["rows"]
    gradient0 = float(ledger["linearisation_reference"]["gradient_norm_at_t0"])
    eps_over_dx = float(ledger["parameters"]["eps"]) / float(ledger["parameters"]["dx"])
    coefficient0_over_dx = float(ledger["linearisation_reference"]["coefficient0_over_dx"])
    guards: dict[str, list[bool]] = {
        "branch_continuity": [],
        "ingredient_continuity": [],
        "step_within_continuity_bound": [],
        "gradient_resolved": [],
        "probe_band": [],
        "arc_open_path": [],
        "ambiguity_free": [],
    }
    for row in rows:
        guards["branch_continuity"].append(row["root_full_status"] == "OK")
        guards["ingredient_continuity"].append(
            all(status == "OK" for status in row["root_ingredient_statuses"].values())
        )
        guards["step_within_continuity_bound"].append(
            row["root_full_over_dx"] is not None
            and abs(float(row["root_full_over_dx"]) - float(row["base_root_over_dx"])) <= readout.ROOT_MATCH_MAX_STEP_DX
        )
        guards["gradient_resolved"].append(
            float(row["gradient_norm_at_point"]) >= readout.GRADIENT_FLOOR_RATIO * gradient0
        )
        guards["probe_band"].append(abs(float(row["phi_at_point"]) - 0.5) <= readout.PROBE_BAND)
        guards["arc_open_path"].append(row["arc_status"] == "OPEN_PATH")
        guards["ambiguity_free"].append(
            row["root_full_status"] not in {"AMBIGUOUS_MULTIPLE_ROOTS", "SADDLE_AMBIGUOUS"}
            and all(
                status not in {"AMBIGUOUS_MULTIPLE_ROOTS", "SADDLE_AMBIGUOUS"}
                for status in row["root_ingredient_statuses"].values()
            )
        )
    failures = {name: int(len(values) - sum(values)) for name, values in guards.items()}
    # the frozen global-contour uniqueness diagnostic of L1A-2x is carried in this stage's ledger and is a
    # guard in its own right: a support point that is equidistant from two contour locations is a
    # non-unique root, and the owner's instruction is that a non-unique root is marked UNRESOLVED
    macro_rows = ledger["macro_rows"]
    uniqueness_records = [row["contour_uniqueness"] for row in macro_rows]
    non_unique_macros = [
        int(row["macro_step"])
        for row, record in zip(macro_rows, uniqueness_records, strict=True)
        if record["status"] != "UNIQUE"
    ]
    global_uniqueness_ok = not non_unique_macros
    macro_guards = {
        "frozen_global_contour_uniqueness": {
            "holds": bool(global_uniqueness_ok),
            "macros_total": len(macro_rows),
            "macros_non_unique": len(non_unique_macros),
            "non_unique_macro_steps": non_unique_macros,
            "statuses_seen": sorted({str(record["status"]) for record in uniqueness_records}),
            "min_margin_over_dx": float(np.min([record["margin_over_dx"] for record in uniqueness_records])),
            "max_margin_over_dx": float(np.max([record["margin_over_dx"] for record in uniqueness_records])),
            "second_candidate_distance_over_dx": sorted(
                {float(record["second_candidate_distance_over_dx"]) for record in uniqueness_records}
            ),
            "rule": (
                "inherited from the frozen L1A-2x contour-uniqueness diagnostic: if the tracked support "
                "point is at the same SDF distance from two contour locations the root at that point is not "
                "unique, and a non-unique root is reported and withholds the verdict. This stage's own "
                "readout guard is the ray-local ambiguity test above; this macro-level guard is reported "
                "because the same non-uniqueness is visible in the fields this stage measured"
            ),
            "source": "ledger macro rows (frozen stage_2x._contour_uniqueness), reported not re-defined",
        }
    }
    equivalence_ok = (
        equivalence["max_phi_difference_vs_eager_public_step"] <= EQUIVALENCE_MAX_PHI_DIFF
        and equivalence["max_phi_difference_vs_jitted_public_step"] <= EQUIVALENCE_MAX_PHI_DIFF
        and (
            equivalence["max_gap_difference_over_dx"] is None
            or equivalence["max_gap_difference_over_dx"] <= EQUIVALENCE_MAX_GAP_DIFF_DX
        )
    )
    terms = residuals["terms_dx"]
    closure_ok = (
        abs(float(terms["R_attribution_closure"])) <= ATTRIBUTION_CLOSURE_BAND_DX
        and abs(float(terms["R_ch_pair_closure"])) <= ATTRIBUTION_CLOSURE_BAND_DX
    )
    nonlinearity_ok = abs(float(terms["R_nonlinearity_point_linearisation"])) <= TOTAL_DISPLACEMENT_CLOSURE_BAND_DX
    static_ok = bool(study["holds"])
    blocking = {name: value for name, value in failures.items() if value > 0}
    if not global_uniqueness_ok:
        blocking["frozen_global_contour_uniqueness"] = len(non_unique_macros)
    comparison_error = max(
        float(attribution["raw_rule_spread_dx"]),
        float(attribution["walk_noise_floor_dx"]),
        WALK_NOISE_FLOOR_DX,
    )
    margin = float(attribution["rule_margins_dx"][attribution["primary_rule"]])
    dominance_floor = DOMINANCE_FACTOR * comparison_error
    return {
        "criteria": {
            "continuity_bound_dx": readout.ROOT_MATCH_MAX_STEP_DX,
            "ambiguity_separation_dx": readout.ROOT_AMBIGUITY_SEPARATION_DX,
            "arc_root_match_tolerance_dx": readout.ARC_ROOT_MATCH_TOLERANCE_DX,
            "gradient_floor_ratio": readout.GRADIENT_FLOOR_RATIO,
            "probe_band": readout.PROBE_BAND,
            "dominance_factor": DOMINANCE_FACTOR,
            "attribution_closure_band_dx": ATTRIBUTION_CLOSURE_BAND_DX,
            "total_displacement_closure_band_dx": TOTAL_DISPLACEMENT_CLOSURE_BAND_DX,
            "static_reconstruction_tolerance_dx": STATIC_RECONSTRUCTION_TOLERANCE_DX,
            "equivalence_max_phi_diff": EQUIVALENCE_MAX_PHI_DIFF,
            "equivalence_max_gap_diff_over_dx": EQUIVALENCE_MAX_GAP_DIFF_DX,
        },
        "substeps": len(rows),
        "failures": failures,
        "macro_guards": macro_guards,
        "blocking_failures": blocking,
        "equivalence_ok": equivalence_ok,
        "attribution_closure_ok": closure_ok,
        "linearisation_closure_ok": nonlinearity_ok,
        "static_reconstruction_ok": static_ok,
        "comparison_error_dx": comparison_error,
        "comparison_error_terms_dx": {
            "rule_spread": float(attribution["raw_rule_spread_dx"]),
            "walk_noise": float(attribution["walk_noise_floor_dx"]),
            "prespecified_floor": WALK_NOISE_FLOOR_DX,
        },
        "dominance_floor_dx": dominance_floor,
        "dominance_margin_dx": margin,
        "unresolved": bool(blocking)
        or not equivalence_ok
        or not closure_ok
        or not nonlinearity_ok
        or not static_ok
        or not global_uniqueness_ok,
        "rule": (
            "a failed guard withholds the verdict, never grants one: a branch that is not continuous, an "
            "ingredient readout that fails, a step beyond the continuity bound, an unresolved local "
            "gradient, a probe point outside the interface band, an arc that is not a simple open path, any "
            "root ambiguity, a non-unique frozen global-contour match at the tracked point, a capture that "
            "does not reproduce the frozen public step, a broken attribution or Cahn-Hilliard-pair closure, a "
            "linearisation error above its closure band, or a static reconstruction beyond its tolerance all "
            "return UNRESOLVED"
        ),
        "observed": {
            "min_gradient_ratio_to_t0": float(np.min([row["gradient_norm_at_point"] for row in rows])) / gradient0,
            "max_abs_probe_offset": float(np.max([abs(row["phi_at_point"] - 0.5) for row in rows])),
            "max_step_over_dx": float(
                np.max(
                    [
                        abs(float(row["root_full_over_dx"]) - float(row["base_root_over_dx"]))
                        for row in rows
                        if row["root_full_over_dx"] is not None
                    ]
                )
            ),
            "max_substep_displacement_over_eps": float(
                np.max(
                    [abs(stage_2x._dn(coefficient0_over_dx, row["dphi_full_at_point"])) / eps_over_dx for row in rows]
                )
            ),
            "segment_binding_switch": ledger["segment_binding_switch"],
            "support_cells_seen": sorted(
                {tuple(row["gap_global_cell"]) for row in rows if row["gap_global_cell"] is not None}, key=str
            ),
            "arc_statuses": sorted({str(row["arc_status"]) for row in rows}),
            "tracked_segment_indices": sorted(
                {int(row["tracked_segment_index"]) for row in rows if row["tracked_segment_index"] is not None}
            ),
        },
    }


def decide(
    ledger: dict[str, Any],
    attribution: dict[str, Any],
    residuals: dict[str, Any],
    study: dict[str, Any],
    validity: dict[str, Any],
    equivalence: dict[str, Any],
) -> dict[str, Any]:
    """One verdict, withheld whenever a prespecified guard fails."""
    del residuals, study, equivalence  # these quantities enter through the validity screen
    margin = float(validity["dominance_margin_dx"])
    floor = float(validity["dominance_floor_dx"])
    comparison_error = float(validity["comparison_error_dx"])
    cumulative = attribution["cumulative_dn_dx"]
    withheld: list[str] = []
    other_blocking = {
        name: value
        for name, value in validity["blocking_failures"].items()
        if name != "frozen_global_contour_uniqueness"
    }
    if other_blocking:
        withheld.append(f"blocking validity guards failed: {json.dumps(other_blocking)}")
    if not validity["equivalence_ok"]:
        withheld.append("the capture does not reproduce the frozen public step within the prespecified thresholds")
    if not validity["attribution_closure_ok"]:
        withheld.append("the attribution closure exceeds its band")
    if not validity["linearisation_closure_ok"]:
        withheld.append("the linearisation error exceeds its closure band")
    if not validity["static_reconstruction_ok"]:
        withheld.append("the static reconstruction exceeds its tolerance")
    if not validity["macro_guards"]["frozen_global_contour_uniqueness"]["holds"]:
        withheld.append(
            "the frozen global contour match at the tracked support point is not unique on "
            f"{validity['macro_guards']['frozen_global_contour_uniqueness']['macros_non_unique']} of "
            f"{validity['macro_guards']['frozen_global_contour_uniqueness']['macros_total']} macro steps"
        )
    if validity["unresolved"] or withheld:
        verdict = "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
        reason = (
            "at least one prespecified guard or gate failed, so no mechanism is claimed; every measurement is "
            "still reported"
        )
    elif margin >= floor:
        verdict = "ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION"
        reason = (
            f"the raw dominance margin {margin:.6g} dx exceeds {DOMINANCE_FACTOR} x the measured comparison "
            f"error {comparison_error:.6g} dx"
        )
    elif -margin >= floor:
        verdict = "CAHN_HILLIARD_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION"
        reason = (
            f"the raw dominance margin {margin:.6g} dx is below minus {DOMINANCE_FACTOR} x the measured "
            f"comparison error {comparison_error:.6g} dx, so the Cahn-Hilliard pair carries the motion"
        )
    else:
        verdict = "MIXED_TERMS_WITHIN_THE_COMPARISON_ERROR"
        reason = (
            f"the difference between the two contributions ({margin:.6g} dx) does not exceed "
            f"{DOMINANCE_FACTOR} x the measured comparison error {comparison_error:.6g} dx"
        )
    return {
        "case_name": ledger["case_name"],
        "verdict": verdict,
        "reason": reason,
        "verdict_withheld_because": withheld,
        "primary_rule": attribution["primary_rule"],
        "cumulative_dn_dx": cumulative,
        "advective_magnitude_dx": abs(float(cumulative["adv"])),
        "net_chemical_hilliard_magnitude_dx": abs(float(cumulative["ch_pair"])),
        "dominance_margin_dx": margin,
        "dominance_floor_dx": floor,
        "dominance_factor": DOMINANCE_FACTOR,
        "comparison_error_dx": comparison_error,
        "comparison_error_terms_dx": validity["comparison_error_terms_dx"],
        "margin_over_comparison_error": (margin / comparison_error) if comparison_error > 0 else None,
        "rule_margins_dx": attribution["rule_margins_dx"],
        "comparison_terms_dx": attribution["comparison_terms_dx"],
        "cahn_hilliard_cancellation_ratio": attribution["cahn_hilliard_pairing"]["cancellation_ratio"],
        "cahn_hilliard_halves_are_one_update": True,
        "segment_binding_switch": ledger["segment_binding_switch"],
        "capture_equivalence_ok": validity["equivalence_ok"],
        "scope": {
            "case": ledger["case_name"],
            "t_window": [0.0, ledger["t_final"]],
            "macro_steps_run": ledger["macro_steps_run"],
            "dt": ledger["requested_dt"],
            "window_limited_to_early_response": True,
            "contact_threshold_reached": ledger["contact_guard"]["triggered"],
            "contact_events_scored": 0,
        },
        "forbidden_claims_respected": {
            "no_contact_mechanism_claim": True,
            "no_impact_claim_from_early_displacement": True,
            "no_extrapolation_to_late_contact_time": True,
            "no_refinement_or_convergence_claim": True,
            "no_production_initializer_promotion": True,
            "no_separate_mechanism_for_the_two_cahn_hilliard_halves": True,
        },
        "next_action_is_not_started_in_this_stage": True,
    }


def next_action(decision: dict[str, Any]) -> dict[str, Any]:
    """Exactly one bounded next action, not started here."""
    if decision["verdict"] == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED":
        withheld = decision.get("verdict_withheld_because", [])
        action = (
            "Prespecify a corrected primary decomposition rule for the identical window - the per-substep "
            "displacement of the field that each increment produces on its own base field, the semantics the "
            "readout design wrote down before this window ran - together with a support-point measurement "
            "that is unique against the frozen global contour, freeze both with their adversarial tests "
            "before any new measurement, and re-run the identical prespecified window with this stage's gates, "
            "bands and constants unchanged. Report the failing quantities of this stage "
            f"({json.dumps(withheld)}) alongside the corrected reading, and claim no mechanism until every "
            "prespecified gate passes."
        )
    else:
        action = (
            "Run the identical prespecified window (one case, one dt, at most 40 macro steps, no extension "
            "toward contact) on the remaining in-band canaries with this stage's readout and criteria "
            "unchanged, and report whether the carrier of the early-window contour motion is case specific."
        )
    return {
        "action": action,
        "bounded_scope": {
            "cases": []
            if decision["verdict"] == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
            else ["flat_we100_ct050", "flat_we200_ct000"],
            "dt": 0.002,
            "macro_step_cap": MAX_MACRO_STEPS,
            "window": "t <= 0.08",
            "solver": "frozen, unmodified",
            "criteria": "this stage's, unchanged",
            "no_extension_toward_contact": True,
            "no_control_arms": True,
            "no_refinement": True,
            "no_generator_or_ml": True,
        },
        "must_not_be_started_in_this_stage": True,
        "requires_owner_approval": True,
    }


# ---------------------------------------------------------------------------
# Section D - cross-check of the identical window against the sealed L1A-2x record
# ---------------------------------------------------------------------------
def _sealed_stage_evidence_unchanged(directory: str) -> dict[str, Any]:
    """Re-hash one frozen stage directory against its own manifest seal (read only)."""
    path = TWO_PHASE / "evidence" / directory
    manifest_path = path / "manifest.json"
    problems: list[str] = []
    if not manifest_path.is_file():
        return {"directory": f"evidence/{directory}", "problems": ["missing manifest.json"], "holds": False}
    manifest = json.loads(manifest_path.read_text())
    seal = stage_2x._recorded_evidence_seal(manifest)
    for name, want in sorted(seal.items()):
        target = path / name
        if not target.is_file():
            problems.append(f"missing bound file {name}")
        elif sha256_file(target) != want:
            problems.append(f"sha mismatch {name}")
    return {
        "directory": f"evidence/{directory}",
        "bound_files": len(seal),
        "problems": problems,
        "holds": bool(seal) and not problems,
    }


def l1a2x_window_cross_check(ledger: dict[str, Any]) -> dict[str, Any]:
    """Compare this stage's window with the sealed L1A-2x window macro step by macro step."""
    sealed_ledger = json.loads((TWO_PHASE / "evidence" / "l1a2x" / "substep_ledger.json").read_text())
    sealed_macro = {int(row["macro_step"]): row for row in sealed_ledger["macro_rows"]}
    rows = []
    for macro_row in ledger["macro_rows"]:
        index = int(macro_row["macro_step"])
        reference = sealed_macro.get(index, {})
        gap_reference = reference.get("gap_global_over_dx")
        level_reference = reference.get("phi_at_support")
        root_reference = reference.get("tracked_edge_y_over_dx")
        line_reference = reference.get("line_root_y_over_dx")
        rows.append(
            {
                "macro_step": index,
                "t": macro_row["t"],
                "gap_this_stage_over_dx": macro_row["gap_global_over_dx"],
                "gap_l1a2x_over_dx": gap_reference,
                "gap_difference_over_dx": (
                    None
                    if gap_reference is None or macro_row["gap_global_over_dx"] is None
                    else abs(float(macro_row["gap_global_over_dx"]) - float(gap_reference))
                ),
                "phi_at_point_this_stage": macro_row["phi_at_point"],
                "phi_at_point_l1a2x": level_reference,
                "phi_at_point_difference": (
                    None if level_reference is None else abs(float(macro_row["phi_at_point"]) - float(level_reference))
                ),
                "root_this_stage_over_dx": macro_row["root_full_over_dx"],
                "root_l1a2x_tracked_edge_over_dx": root_reference,
                "root_difference_over_dx": (
                    None
                    if root_reference is None or macro_row["root_full_over_dx"] is None
                    else abs(float(macro_row["root_full_over_dx"]) - float(root_reference))
                ),
                "root_l1a2x_was_lost": bool(reference.get("tracked_edge_branch_lost", False)),
                "line_root_l1a2x_over_dx": line_reference,
                "line_root_difference_over_dx": (
                    None
                    if line_reference is None or macro_row["root_full_over_dx"] is None
                    else abs(float(macro_row["root_full_over_dx"]) - float(line_reference))
                ),
                "segment_index_this_stage": macro_row["tracked_segment_index"],
            }
        )
    gap_differences = [row["gap_difference_over_dx"] for row in rows if row["gap_difference_over_dx"] is not None]
    level_differences = [row["phi_at_point_difference"] for row in rows if row["phi_at_point_difference"] is not None]
    root_differences = [
        row["root_difference_over_dx"]
        for row in rows
        if row["root_difference_over_dx"] is not None and not row["root_l1a2x_was_lost"]
    ]
    rows_l1a2x_could_measure = [row for row in rows if not row["root_l1a2x_was_lost"]]
    line_differences = [
        row["line_root_difference_over_dx"] for row in rows if row["line_root_difference_over_dx"] is not None
    ]
    lost = [row["macro_step"] for row in rows if row["root_l1a2x_was_lost"]]
    final_state_matches = {
        key: ledger["final_state"][f"{key}_hash"] == sealed_ledger["final_state"][f"{key}_hash"]
        for key in ("phi", "u", "v")
    }
    switch = ledger["segment_binding_switch"]
    switch_macro = switch.get("macro_step")
    reference_t = None
    if switch_macro is not None:
        reference_t = sealed_macro.get(int(switch_macro), {}).get("t")
    checks = {
        "the_window_is_the_identical_one": bool(
            len(rows) == len(sealed_macro)
            and gap_differences
            and max(gap_differences) <= 1.0e-12
            and level_differences
            and max(level_differences) <= 1.0e-12
        ),
        "final_state_hashes_match_the_sealed_window": all(final_state_matches.values()),
        "the_support_change_of_l1a2x_is_identified_here": bool(switch["found"] and switch_macro is not None),
        "the_support_change_is_reported_as_a_continuous_event": bool(
            switch["found"]
            and switch_macro is not None
            and (not lost or int(switch_macro) == int(lost[0]))
            and reference_t is not None
            and abs(float(switch["t"]) - float(reference_t)) <= 1.0e-12
        ),
        "root_agrees_where_l1a2x_could_measure_it": bool(
            len(rows_l1a2x_could_measure) == len(sealed_macro) - len(lost)
            and root_differences
            and max(root_differences) <= CROSS_CHECK_TOLERANCE_DX
        ),
        "this_stage_measures_the_window_l1a2x_could_not": bool(
            len(rows) == len(sealed_macro)
            and lost
            and all(row["root_this_stage_over_dx"] is not None for row in rows)
            and all(row["line_root_l1a2x_over_dx"] is not None for row in rows)
        ),
        "the_independent_line_root_agrees_with_the_tracked_root": bool(
            line_differences and max(line_differences) <= CROSS_CHECK_TOLERANCE_DX
        ),
    }
    return {
        "stage": STAGE,
        "source": "evidence/l1a2x (sealed, unedited): substep_ledger.json",
        "rows": rows,
        "max_gap_difference_over_dx": max(gap_differences) if gap_differences else None,
        "max_phi_at_point_difference": max(level_differences) if level_differences else None,
        "max_root_difference_over_dx": max(root_differences) if root_differences else None,
        "max_line_root_difference_over_dx": max(line_differences) if line_differences else None,
        "macros_where_l1a2x_lost_its_pinned_edge": lost,
        "macros_where_l1a2x_could_measure_its_pinned_edge": len(rows_l1a2x_could_measure),
        "first_lost_macro_step": lost[0] if lost else None,
        "macros_measured_by_each_stage": {"this_stage": len(rows), "l1a2x": len(sealed_macro)},
        "support_change_this_stage": switch,
        "final_state_hash_matches": final_state_matches,
        "tolerance_over_dx": CROSS_CHECK_TOLERANCE_DX,
        "checks": checks,
        "all_hold": all(checks.values()),
        "rule": (
            "this stage's window must be the identical trajectory: the frozen gap and the level at the pinned "
            "point must reproduce the sealed L1A-2x values at every macro step and the final state must hash to "
            "the sealed final state; the macro step at which L1A-2x lost its pinned edge must be reported here "
            "as a continuous support change, the tracked root must agree with the L1A-2x line root where that "
            "stage could measure it, and this stage must still measure the steps L1A-2x lost"
        ),
    }


# ---------------------------------------------------------------------------
# Section E - adversarial and falsification checks
# ---------------------------------------------------------------------------
def _fabricated_validity(validity: dict[str, Any], **updates: Any) -> dict[str, Any]:
    """A copy of a validity screen with chosen fields overridden (for exercising the decision rule).

    The fabricated screens leave every gate that this stage did not fail in a passing state, so that the
    decision rule itself is exercised and not the measured screen.
    """
    healthy: dict[str, Any] = {
        "blocking_failures": {},
        "equivalence_ok": True,
        "attribution_closure_ok": True,
        "linearisation_closure_ok": True,
        "static_reconstruction_ok": True,
        "unresolved": False,
    }
    healthy.update(updates)
    clone: dict[str, Any] = json.loads(json.dumps(validity))
    clone.update(healthy)
    macro_guards = clone.get("macro_guards") or {}
    uniqueness = macro_guards.get("frozen_global_contour_uniqueness")
    if uniqueness is not None and "uniqueness_holds" not in updates:
        uniqueness["holds"] = True
        uniqueness["macros_non_unique"] = 0
        uniqueness["non_unique_macro_steps"] = []
    return clone


def adversarial_and_falsification(
    ledger: dict[str, Any],
    attribution: dict[str, Any],
    residuals: dict[str, Any],
    study: dict[str, Any],
    validity: dict[str, Any],
    decision: dict[str, Any],
    provenance: dict[str, Any],
    cross_check: dict[str, Any],
    reaudit: dict[str, Any],
    equivalence_payload: dict[str, Any],
) -> dict[str, Any]:
    """Falsification attempts: every claim of this stage must be able to fail."""
    bundle, state0 = stage_2x._bundle_and_state()
    p = bundle.p
    phi0 = np.asarray(state0.phi, dtype=np.float64)
    support = stage_2x.fixed_support(bundle, state0)
    spec = readout.branch_definition(p, phi0, support)
    pinned = float(spec["t0_root_over_dx"])
    synthetic = readout.synthetic_validation_report()

    # 1. the support change that stopped L1A-2x, exercised on the real grid: a crossing walks from one cell
    #    centre to the next, is reported once where it lands exactly on a cell centre, and changes its
    #    binding segment exactly once without ever being lost
    walk_rows = []
    tracked = pinned
    for step in range(1, 8):
        height = 23.3 + 0.2 * step
        field = readout.diffuse_column_field(p, crossing_over_dx=height, eps_over_dx=1.5)
        record = readout.branch_root(field, spec, tracked, with_arc=False)
        matched = record["matched"]
        walk_rows.append(
            {
                "applied_height_over_dx": height,
                "status": record["status"],
                "root_over_dx": None if matched is None else float(matched["y_over_dx"]),
                "segment_index": None if matched is None else matched["segment_index"],
                "at_shared_node": None if matched is None else bool(matched["at_shared_node"]),
                "step_over_dx": record["step_over_dx"],
            }
        )
        if matched is not None:
            tracked = float(matched["y_over_dx"])
    segments = [row["segment_index"] for row in walk_rows]
    switches = [index for index in range(1, len(segments)) if segments[index] != segments[index - 1]]
    steps = [row["step_over_dx"] for row in walk_rows if row["step_over_dx"] is not None]
    # 2. the same field: a crossing exactly on a cell centre is reported once, and a jump beyond the
    #    continuity bound is refused
    node_field = readout.diffuse_column_field(p, crossing_over_dx=24.5, eps_over_dx=1.5)
    node_record = readout.branch_root(node_field, spec, 24.4, with_arc=False)
    jump_record = readout.branch_root(node_field, spec, 24.02, with_arc=False)
    # 3. the reader is direction blind on the real field: mirroring the wall normal flips the sign exactly
    last_root = float(ledger["macro_rows"][-1]["root_full_over_dx"])
    wall_normal = tuple(float(value) for value in spec["wall_normal_into_solid"])
    forward = _wall_normal_displacement(pinned, last_root, wall_normal)
    mirrored = _wall_normal_displacement(pinned, last_root, (0.0, -wall_normal[1]))
    # 4. the prespecified decision rule, exercised at its own boundaries with fabricated screens
    floor = float(validity["dominance_floor_dx"])
    fabricated = {
        "blocking_guard_failed": decide(
            ledger,
            attribution,
            residuals,
            study,
            _fabricated_validity(validity, blocking_failures={"branch_continuity": 1}, unresolved=True),
            equivalence_payload,
        )["verdict"],
        "margin_below_the_floor": decide(
            ledger,
            attribution,
            residuals,
            study,
            _fabricated_validity(validity, blocking_failures={}, unresolved=False, dominance_margin_dx=0.999 * floor),
            equivalence_payload,
        )["verdict"],
        "margin_at_the_floor": decide(
            ledger,
            attribution,
            residuals,
            study,
            _fabricated_validity(validity, blocking_failures={}, unresolved=False, dominance_margin_dx=floor),
            equivalence_payload,
        )["verdict"],
        "margin_at_minus_the_floor": decide(
            ledger,
            attribution,
            residuals,
            study,
            _fabricated_validity(validity, blocking_failures={}, unresolved=False, dominance_margin_dx=-floor),
            equivalence_payload,
        )["verdict"],
        "non_unique_global_contour": decide(
            ledger,
            attribution,
            residuals,
            study,
            _fabricated_validity(
                validity,
                blocking_failures={"frozen_global_contour_uniqueness": 40},
                unresolved=True,
                uniqueness_holds=False,
            ),
            equivalence_payload,
        )["verdict"],
    }
    # 5. a fabricated violation of the attribution band must be caught by the closure gate. The observed
    #    closure of this stage is a result, it is reported here and in the residual decomposition, and it is
    #    never forced inside the band by this check
    observed_closure = float(residuals["terms_dx"]["R_attribution_closure"])
    healthy_closure = abs(observed_closure) <= ATTRIBUTION_CLOSURE_BAND_DX
    inside = json.loads(json.dumps(residuals))
    inside["terms_dx"]["R_attribution_closure"] = 0.5 * ATTRIBUTION_CLOSURE_BAND_DX
    inside["terms_dx"]["R_ch_pair_closure"] = 0.5 * ATTRIBUTION_CLOSURE_BAND_DX
    beyond = json.loads(json.dumps(residuals))
    beyond["terms_dx"]["R_attribution_closure"] = 10.0 * ATTRIBUTION_CLOSURE_BAND_DX
    inside_screen = validity_screen(ledger, attribution, inside, study, equivalence_payload)
    beyond_screen = validity_screen(ledger, attribution, beyond, study, equivalence_payload)
    closure_gate_discriminates = bool(
        inside_screen["attribution_closure_ok"] and not beyond_screen["attribution_closure_ok"]
    )
    observed_own_base_closure = float(residuals["terms_dx"]["R_attribution_closure_own_base"])
    # 6. the comparison error must be a raw spread, not a rescaled one
    margins = attribution["rule_margins_dx"]
    primary_margin = float(margins[attribution["primary_rule"]])
    raw_terms = {
        name: float(value) - primary_margin for name, value in margins.items() if name != attribution["primary_rule"]
    }
    raw_spread = max(abs(value) for value in raw_terms.values())
    comparison_is_raw = bool(
        raw_terms
        and all(abs(raw_terms[name] - float(attribution["comparison_terms_dx"][name])) <= 1.0e-12 for name in raw_terms)
        and abs(raw_spread - float(attribution["raw_rule_spread_dx"])) <= 1.0e-12
        and raw_spread > 1.0e-6
    )
    # 7. the L1A-2x spread must be reproduced from its own sealed payload
    family = json.loads((TWO_PHASE / "evidence" / "l1a2x" / "interface_attribution.json").read_text())["family"]
    x2_primary = float(family["PRIMARY_POINT_T0_COEFFICIENT"]["margin_dx"])
    x2_spread = max(
        abs(float(entry["margin_dx"]) - x2_primary)
        for name, entry in family.items()
        if name != "PRIMARY_POINT_T0_COEFFICIENT"
    )
    # 7b. the arithmetic rules of this stage must reproduce the sealed L1A-2x arithmetic of the same window
    sealed_family = json.loads((TWO_PHASE / "evidence" / "l1a2x" / "interface_attribution.json").read_text())["family"]
    sealed_point = sealed_family["PRIMARY_POINT_T0_COEFFICIENT"]["cumulative_dn_dx"]
    sealed_cell = sealed_family["SUPPORT_CELL_T0_COEFFICIENT"]["cumulative_dn_dx"]
    linearised = attribution["linearised_dn_dx"]
    linearised_cell = attribution["linearised_cell_dn_dx"]
    arithmetic_reproduction = {
        "point": {key: abs(float(linearised[key]) - float(sealed_point[key])) for key in ("adv", "ch", "imp", "full")},
        "support_cell": {
            key: abs(float(linearised_cell[key]) - float(sealed_cell[key])) for key in ("adv", "ch", "imp", "full")
        },
    }
    arithmetic_reproduced = all(
        value <= 1.0e-12 for table in arithmetic_reproduction.values() for value in table.values()
    )
    # 7c. the static reconstruction gate must also discriminate when it is violated
    broken_study = json.loads(json.dumps(study))
    broken_study["holds"] = False
    broken_study["max_static_reconstruction_forward_over_dx"] = 10.0 * STATIC_RECONSTRUCTION_TOLERANCE_DX
    broken_screen = validity_screen(ledger, attribution, residuals, broken_study, equivalence_payload)
    static_gate_bites = bool(
        study["holds"] and not broken_screen["static_reconstruction_ok"] and broken_screen["unresolved"]
    )
    # 7d. the reported attribution must be a pure function of the recorded ledger
    recomputed_attribution = interface_attribution(ledger)
    attribution_is_reproducible = bool(
        json.dumps(recomputed_attribution, sort_keys=True, default=_json_default)
        == json.dumps(attribution, sort_keys=True, default=_json_default)
    )
    # 7e. the withheld verdict must not be turnable into a mechanism claim by choosing another rule
    nicer_attribution = json.loads(json.dumps(attribution))
    nicer_attribution["primary_rule"] = "RULE_B_GEOMETRIC_OWN_BASE"
    nicer_screen = json.loads(json.dumps(validity))
    nicer_screen["dominance_margin_dx"] = float(attribution["rule_margins_dx"]["RULE_B_GEOMETRIC_OWN_BASE"])
    nicer_decision = decide(ledger, nicer_attribution, residuals, study, nicer_screen, equivalence_payload)
    nicer_rule_cannot_flip_the_verdict = bool(
        nicer_decision["verdict"] == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
        and nicer_decision["verdict_withheld_because"]
    )
    # 8. the frozen sources, the readout module and the sealed directories must be unmodified
    frozen_hashes_hold = all(
        want == sha256_file(REPO_ROOT / name) for name, want in provenance["frozen_source_sha256"].items()
    )
    readout_hash_holds = provenance["readout_module_sha256"] == sha256_file(READOUT_MODULE)
    sealed_checks = {name: _sealed_stage_evidence_unchanged(name) for name in ("l1a2x", "l1a2w")}
    model_flags = {
        "solver_contract_version": (int(pf.SOLVER_CONTRACT_VERSION), int(provenance["solver_contract_version"])),
        "phase_storage_model": (str(p.phase_storage_model), str(provenance["phase_storage_model"])),
        "phase_transport_geometry": (str(p.phase_transport_geometry), str(provenance["phase_transport_geometry"])),
        "phase_boundary_model": (str(p.phase_boundary_model), str(provenance["phase_boundary_model"])),
        "l1a_status": ("BLOCKED", "BLOCKED"),
        "l1b_data_not_ready": ("unchanged", str(provenance["frozen_status_flags"]["L1B_DATA_NOT_READY"])),
    }
    model_flags_hold = all(observed == recorded for observed, recorded in model_flags.values())
    # 9. the window budget must bite before any solver call
    capped = False
    try:
        run_window(MAX_MACRO_STEPS + 1)
    except AuditBlocked:
        capped = True
    # 10. the ledger must not be vacuous
    non_vacuous = {
        "substeps": int(len(ledger["rows"])),
        "max_abs_mass_adv": float(np.max(np.abs([row["mass_adv"] for row in ledger["rows"]]))),
        "max_abs_mass_ch": float(np.max(np.abs([row["mass_ch"] for row in ledger["rows"]]))),
        "max_abs_mass_imp": float(np.max(np.abs([row["mass_imp"] for row in ledger["rows"]]))),
        "macros_with_a_resolved_support_cell": int(study["macros_with_a_resolved_support_cell"]),
        "tracked_segments_seen": validity["observed"]["tracked_segment_indices"],
    }
    # 11. the reader must not be able to pick a contour that favours the target direction
    favourable = synthetic["cases"]["no_favourable_selection"]
    # 12. no forbidden claim may appear in the reported payload
    forbidden = (
        "real impact",
        "impact mechanism",
        "confirmed contact",
        "contact event occurred",
        "two independent mechanisms",
    )
    reported_text = json.dumps(
        {"decision": decision, "study": study, "attribution": attribution}, default=_json_default
    )
    found_forbidden = [phrase for phrase in forbidden if phrase in reported_text]
    checks = {
        "reader_tracks_the_support_change_on_the_real_grid": {
            "rows": walk_rows,
            "segment_indices": segments,
            "segment_switches_at": switches,
            "max_step_over_dx": max(steps) if steps else None,
            "holds": bool(
                all(row["status"] == "OK" for row in walk_rows)
                and segments[0] == 23
                and segments[-1] == 24
                and len(switches) == 1
                and steps
                and max(steps) <= readout.ROOT_MATCH_MAX_STEP_DX
            ),
        },
        "a_crossing_exactly_on_a_cell_centre_is_reported_once": {
            "status": node_record["status"],
            "root_over_dx": None if node_record["matched"] is None else node_record["matched"]["y_over_dx"],
            "at_shared_node": None
            if node_record["matched"] is None
            else bool(node_record["matched"]["at_shared_node"]),
            "holds": bool(
                node_record["status"] == "OK"
                and node_record["matched"] is not None
                and bool(node_record["matched"]["at_shared_node"])
                and abs(float(node_record["matched"]["y_over_dx"]) - 24.5) <= 1.0e-9
            ),
        },
        "a_jump_beyond_the_continuity_bound_is_refused_on_the_real_grid": {
            "status": jump_record["status"],
            "step_over_dx": jump_record["step_over_dx"],
            "continuity_bound_over_dx": readout.ROOT_MATCH_MAX_STEP_DX,
            "reason": jump_record["reason"],
            "holds": bool(
                jump_record["status"] == "BRANCH_STEP_TOO_LARGE"
                and jump_record["step_over_dx"] is not None
                and float(jump_record["step_over_dx"]) > readout.ROOT_MATCH_MAX_STEP_DX
                and "continuity bound" in str(jump_record["reason"])
            ),
        },
        "reader_is_direction_blind_on_the_real_field": {
            "displacement_forward_over_dx": forward,
            "displacement_mirrored_over_dx": mirrored,
            "holds": bool(
                forward is not None and mirrored is not None and abs(float(forward) + float(mirrored)) <= 1.0e-12
            ),
        },
        "the_reader_cannot_pick_a_favourable_root": {
            "case": favourable,
            "holds": bool(
                favourable["prefers_the_nearer_root"] and favourable["mirrored_wall_normal_gives_the_same_root"]
            ),
        },
        "the_decision_rule_follows_its_prespecified_thresholds": {
            "fabricated_screens": fabricated,
            "holds": bool(
                fabricated["blocking_guard_failed"] == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
                and fabricated["margin_below_the_floor"] == "MIXED_TERMS_WITHIN_THE_COMPARISON_ERROR"
                and fabricated["margin_at_the_floor"] == "ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION"
                and fabricated["margin_at_minus_the_floor"]
                == "CAHN_HILLIARD_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION"
            ),
        },
        "attribution_closure_gate_bites": {
            "band_over_dx": ATTRIBUTION_CLOSURE_BAND_DX,
            "observed_closure_over_dx": observed_closure,
            "observed_closure_within_band": healthy_closure,
            "observed_closure_over_band_ratio": abs(observed_closure) / ATTRIBUTION_CLOSURE_BAND_DX,
            "observed_own_base_closure_over_dx": observed_own_base_closure,
            "observed_own_base_closure_within_band": abs(observed_own_base_closure) <= ATTRIBUTION_CLOSURE_BAND_DX,
            "fabricated_closure_inside_the_band": 0.5 * ATTRIBUTION_CLOSURE_BAND_DX,
            "fabricated_closure_beyond_the_band": 10.0 * ATTRIBUTION_CLOSURE_BAND_DX,
            "gate_accepts_the_fabricated_inside_band_closure": inside_screen["attribution_closure_ok"],
            "gate_rejects_the_fabricated_beyond_band_closure": not beyond_screen["attribution_closure_ok"],
            "holds": closure_gate_discriminates,
            "rule": (
                "the falsification is that the closure gate discriminates: the same residual decomposition "
                "with a fabricated closure inside the band is accepted and with a fabricated closure beyond "
                "the band it is rejected. The observed closure of this stage is a result; it is reported "
                "here, in the residual decomposition and in the validity screen, and it is never forced "
                "inside the band"
            ),
        },
        "comparison_error_is_not_normalised": {
            "raw_terms_over_dx": raw_terms,
            "recorded_terms_over_dx": attribution["comparison_terms_dx"],
            "raw_spread_over_dx": raw_spread,
            "recorded_spread_over_dx": attribution["raw_rule_spread_dx"],
            "holds": comparison_is_raw,
        },
        "l1a2x_spread_is_reproduced_from_its_sealed_payload": {
            "recomputed_spread_over_dx": x2_spread,
            "recorded_in_the_reaudit_over_dx": reaudit["raw_margin_spread_dx"],
            "recorded_by_l1a2x_itself_over_dx": reaudit["how_l1a2x_reported_it"][
                "recorded_margin_spread_across_rules_dx"
            ],
            "holds": bool(abs(x2_spread - float(reaudit["raw_margin_spread_dx"])) <= 1.0e-15),
        },
        "the_arithmetic_rules_reproduce_the_sealed_l1a2x_window": {
            "max_differences_over_dx": arithmetic_reproduction,
            "holds": bool(arithmetic_reproduced),
            "rule": (
                "the point-linearised and support-cell-linearised rules of this stage rebuild the two "
                "arithmetic rules of the sealed L1A-2x stage on the identical window; every ingredient and the "
                "complete update must agree with the sealed numbers to round-off"
            ),
        },
        "the_static_reconstruction_gate_bites_when_it_is_violated": {
            "observed_holds": study["holds"],
            "fabricated_deviation_over_dx": 10.0 * STATIC_RECONSTRUCTION_TOLERANCE_DX,
            "gate_rejects_the_fabricated_study": not broken_screen["static_reconstruction_ok"],
            "holds": static_gate_bites,
            "rule": (
                "the observed static reconstruction holds; the same screen fed a fabricated study beyond its "
                "tolerance is rejected and returns unresolved, so the gate this stage reports as passing is "
                "demonstrably able to fail"
            ),
        },
        "the_attribution_is_a_pure_function_of_the_recorded_ledger": {
            "holds": attribution_is_reproducible,
            "rule": (
                "recomputing the whole attribution from the recorded substep ledger reproduces the reported "
                "payload exactly, so every number of this stage is a function of the one frozen window and "
                "not of any later choice"
            ),
        },
        "the_withheld_verdict_cannot_be_flipped_by_choosing_a_nicer_rule": {
            "fabricated_primary_rule": "RULE_B_GEOMETRIC_OWN_BASE",
            "fabricated_margin_over_dx": float(attribution["rule_margins_dx"]["RULE_B_GEOMETRIC_OWN_BASE"]),
            "resulting_verdict": nicer_decision["verdict"],
            "resulting_withheld_reasons": nicer_decision["verdict_withheld_because"],
            "holds": nicer_rule_cannot_flip_the_verdict,
            "rule": (
                "the failing gates of this stage are properties of the measurement, not of the chosen rule: "
                "re-running the prespecified decision rule with the alternative primary rule and that rule's "
                "own margin still returns UNRESOLVED, so no mechanism verdict is obtainable in this stage by "
                "selecting a rule after seeing the window"
            ),
        },
        "frozen_sources_and_the_readout_are_unmodified": {
            "frozen_hashes_hold": frozen_hashes_hold,
            "readout_hash_holds": readout_hash_holds,
            "readout_module_sha256": provenance["readout_module_sha256"],
            "contract_and_model_flags": model_flags,
            "holds": bool(frozen_hashes_hold and readout_hash_holds and model_flags_hold),
        },
        "the_sealed_stages_are_untouched_on_disk": {
            "stages": sealed_checks,
            "holds": bool(all(entry["holds"] for entry in sealed_checks.values())),
        },
        "the_window_budget_is_capped": {
            "refused_a_request_above_the_cap": capped,
            "cap_macro_steps": MAX_MACRO_STEPS,
            "holds": capped,
        },
        "the_ledger_is_not_vacuous": {
            **non_vacuous,
            "holds": bool(non_vacuous["max_abs_mass_ch"] > 0.0 and non_vacuous["substeps"] > 0),
        },
        "identical_window_cross_check_holds": {
            "all_hold": cross_check["all_hold"],
            "checks": cross_check["checks"],
            "max_gap_difference_over_dx": cross_check["max_gap_difference_over_dx"],
            "first_lost_macro_step_of_l1a2x": cross_check["first_lost_macro_step"],
            "holds": bool(cross_check["all_hold"]),
        },
        "no_forbidden_claim_in_the_reported_payload": {
            "forbidden_phrases": list(forbidden),
            "found": found_forbidden,
            "holds": not found_forbidden,
        },
    }
    failed = [name for name, check in checks.items() if not bool(check.get("holds", False))]
    return {
        "stage": STAGE,
        "identical_window_cross_check": {key: value for key, value in cross_check.items() if key != "rows"},
        "checks": checks,
        "all_hold": not failed,
        "failed_checks": failed,
        "rule": (
            "every claim of this stage is exercised so that it can fail: the reader is walked across the very "
            "support change that stopped L1A-2x on the real grid, a crossing that lands exactly on a cell centre "
            "is checked to be reported once, a jump beyond the continuity bound is checked to be refused, the "
            "wall normal is mirrored to show the reader has no direction preference, the prespecified decision "
            "rule is exercised at its own thresholds with fabricated validity screens, the closure gate is "
            "exercised against a fabricated violation, the comparison error is checked to be a raw spread rather "
            "than a rescaled one, the L1A-2x spread is recomputed from its sealed payload, the frozen sources and "
            "the readout module are re-hashed, the sealed evidence directories are re-hashed against their own "
            "manifests, the window cap is exercised, and the reported payload is searched for forbidden claims"
        ),
    }


# ---------------------------------------------------------------------------
# Section F - quality, focused tests and hosted CI
# ---------------------------------------------------------------------------
FROZEN_ADJACENT_FAILING_SET = (
    "test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight",
    "test_interior_speed_admissibility_screen_audit.py::test_required_evidence_files_are_declared_and_complete",
    "test_interior_speed_admissibility_screen_audit.py::test_manifest_self_verification_and_tamper_detection",
    "test_interior_speed_admissibility_screen_audit.py::test_evidence_verification_rejects_a_moved_envelope",
    "test_interior_speed_admissibility_screen_audit.py::test_reseal_records_the_observed_ci_status_without_touching_physics",
    "test_interior_speed_admissibility_screen_audit.py::test_report_carries_every_required_section",
    "test_boundary_compatible_incident_momentum_bound_audit.py::test_preflight_is_clean_on_the_repository_head",
    "test_radial_incident_seed_eta_dt_audit.py::test_report_contains_every_required_section",
)
FROZEN_ADJACENT_SUITE_FAILURES = {"l1a2t": 1, "l1a2u": 1, "l1a2v": 5, "l1a2w": 1, "l1a2x": 0}
LEGACY_LINEAGE_FAILING_SET = (
    "test_impact_initialization_compatibility_audit.py::test_l1a2r_merged_base_verified_or_fail_closed",
)
FROZEN_SWEEP_COMMAND = (
    "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest "
    "tests/test_nearwall_taper_interface_transport_audit.py "
    "tests/test_radial_incident_seed_eta_dt_audit.py "
    "tests/test_interior_speed_admissibility_screen_audit.py "
    "tests/test_boundary_compatible_incident_momentum_bound_audit.py "
    "tests/test_frozen_chns_early_window_transport_attribution_audit.py -q"
)
COMBINED_SWEEP_COMMAND = (
    "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest "
    "tests/test_nearwall_taper_interface_transport_audit.py "
    "tests/test_radial_incident_seed_eta_dt_audit.py "
    "tests/test_interior_speed_admissibility_screen_audit.py "
    "tests/test_boundary_compatible_incident_momentum_bound_audit.py "
    "tests/test_frozen_chns_early_window_transport_attribution_audit.py "
    "tests/test_impact_initialization_compatibility_audit.py -q"
)
LEGACY_SWEEP_COMMAND = (
    "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest "
    "tests/test_impact_initialization_compatibility_audit.py -q"
)


def _frozen_sweep_measured_at() -> str:
    """Where the recorded sweep was taken: the head at run time plus the worktree state it saw.

    The head is resolved when the payload is written instead of being typed in, so the recorded state can
    never drift away from the commit the stage actually ran at.
    """
    head = _git(["rev-parse", "HEAD"], required=False) or "UNKNOWN_HEAD"
    return (
        f"head {head}, with this stage's own evidence directory present in the worktree but uncommitted "
        "(the state the stage produced that evidence in)"
    )


def quality_and_blockers(verdict: str) -> dict[str, Any]:
    """Declared quality expectation, the frozen adjacent suites and the CI bookkeeping."""
    baseline = REPO_ROOT / "quality" / "legacy_two_phase_debt_v1.json"
    return {
        "stage": STAGE,
        "declared_new_code_diagnostics": 0,
        "quality_baseline_path": "quality/legacy_two_phase_debt_v1.json",
        "quality_baseline_sha256": sha256_file(baseline) if baseline.is_file() else "MISSING",
        "ratchet_expectation": "PASS_BASELINE_ONLY",
        "legacy_debt_baseline": {"ruff": 171, "formatter": 27, "isort": 28, "codespell": 3},
        "hosted_ci": {
            "status_for_this_head": "UNMEASURED",
            "observed_head_sha": None,
            "rule": "no hosted result is claimed for a head that was not pushed and observed",
        },
        "frozen_adjacent_suite": {
            "classification": "SUPERSEDED_BY_THE_NEXT_STAGE_BY_CONSTRUCTION",
            "suite_failures": dict(FROZEN_ADJACENT_SUITE_FAILURES),
            "suite_failure_count": int(sum(FROZEN_ADJACENT_SUITE_FAILURES.values())),
            "failing": list(FROZEN_ADJACENT_FAILING_SET),
            "sweep_command": FROZEN_SWEEP_COMMAND,
            "sweep_result": "8 failed, 221 passed",
            "measured_at": _frozen_sweep_measured_at(),
            "state_dependence": (
                "measured in two states of the same head. With this stage's own evidence directory present "
                "in the worktree but uncommitted - the state this stage produced its evidence in - the five "
                "suites report 8 failures, including the L1A-2u report-section check, because that preflight "
                "refuses to build its state while the worktree is dirty outside its own paths. With that "
                "directory absent the same five suites report 7 and that check passes. The set below is the "
                "one observed in the state recorded in preflight.json, which is the state the stage ran in"
            ),
            "combined_sweep_observation": {
                "command": COMBINED_SWEEP_COMMAND,
                "result": "9 failed, 269 passed",
                "split": (
                    "the eight structural failures listed here plus the one legacy lineage failure, which "
                    "is reported separately below"
                ),
                "derivation": (
                    "the five predecessor suites returned 8 failed / 221 passed, the lineage suite returned "
                    "1 failed / 48 passed, and one invocation of the combined command returned 9 failed / "
                    "269 passed; all three were runs in this state and the combined figure is the observed "
                    "one, not a reconstruction"
                ),
            },
            "cause": (
                "the frozen predecessor preflights require every commit above their own baseline to belong "
                "to their own stage, and two of them additionally require a worktree clean outside their own "
                "paths; neither can hold once a later stage is committed on the same branch, so these "
                "failures are structural and are recorded as observed rather than repaired. Nothing here is "
                "an arithmetic or physics disagreement with a predecessor result"
            ),
            "action": "recorded truthfully as observed; no frozen test was edited and none was turned green",
        },
        "legacy_lineage_suite": {
            "failing": list(LEGACY_LINEAGE_FAILING_SET),
            "count": len(LEGACY_LINEAGE_FAILING_SET),
            "sweep_command": LEGACY_SWEEP_COMMAND,
            "sweep_result": "1 failed, 48 passed",
            "classification": "FROZEN_LINEAGE_DRIFT_CONFIRMED",
            "cause": "pre-existing; reproduced on a pristine worktree of the verified main, not caused by this stage",
            "action": "kept strictly separate from the structural failures and never counted as a pass",
        },
        "reporting_rule": (
            "the structural frozen-adjacent failures and the old lineage failure are reported separately with "
            "their own counts; nothing is hidden, suppressed or forced green"
        ),
        "stage_test_file": "tests/test_frozen_reconstruction_branch_attribution_audit.py",
        "verdict": verdict,
        "not_run": [
            "control trajectories",
            "contact events",
            "temporal refinement",
            "spatial refinement",
            "T=8 recomputation",
            "L1B data generation",
            "ML training",
            "contract 13",
            "parameter sweep",
        ],
    }


# ---------------------------------------------------------------------------
# Section G - the report
# ---------------------------------------------------------------------------
def render_report(result: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Render the human-readable report from the assembled payloads."""
    ledger = result["substep_ledger.json"]
    design = result["readout_design.json"]
    synthetic = result["synthetic_reader_validation.json"]
    frozen_equivalence = result["frozen_reconstruction_equivalence.json"]
    reaudit = result["comparison_error_reaudit.json"]
    identity = result["initial_state_identity.json"]
    attribution = result["interface_attribution.json"]
    study = result["static_reconstruction_study.json"]
    residuals = result["residual_decomposition.json"]
    conservation = result["conservation_and_convergence.json"]
    equivalence = result["capture_equivalence.json"]
    validity = result["validity_screen.json"]
    decision = result["mechanism_decision.json"]
    action = result["next_action.json"]
    adversarial = result["adversarial_and_falsification.json"]
    quality = result["quality_and_blocker_matrix.json"]
    provenance = result["source_and_provenance.json"]
    prerequisites = result["frozen_evidence_verification.json"]
    cross_check = adversarial["identical_window_cross_check"]
    diagnosis = residuals["post_window_diagnosis"]
    cumulative = attribution["cumulative_dn_dx"]
    bias_case = synthetic["cases"]["reconstruction_bias_is_measured_not_assumed"]
    lines: list[str] = []
    add = lines.append
    add(f"# {STAGE} - reconstruction-stable early-window interface transport attribution")
    add("")
    add("## A. Scope, authority and baseline")
    add(
        f"- baseline: PR #24 head `{BASELINE_HEAD_SHA}`; verified main `{VERIFIED_MAIN_SHA}`; branch "
        f"`{provenance['branch']}`; solver contract {provenance['solver_contract_version']}; boundary model "
        f"`{provenance['phase_boundary_model']}`"
    )
    add(
        f"- one case `{ledger['case_name']}`, N={ledger['profile_N']}, dt={ledger['requested_dt']}, "
        f"{ledger['macro_steps_run']} macro steps of at most {MAX_MACRO_STEPS} "
        f"({ledger['substeps_recorded']} frozen substeps, t <= {_fmt(ledger['t_final'], 6)})"
    )
    add(
        f"- trajectory count 1, control arms {ledger['prohibitions']['control_arm_runs']}, contact events "
        f"scored {ledger['prohibitions']['contact_events_scored']}, refinements "
        f"{ledger['prohibitions']['refinement_runs']}, T=8 recomputed {ledger['prohibitions']['t8_recomputed']}, "
        f"L1B data generated {ledger['prohibitions']['l1b_data_generated']}"
    )
    add("")
    add("## B. Phase 1: the prespecified readout")
    add(f"- readout version `{design['readout_version']}`, fingerprint `{design['fingerprint_sha256'][:16]}...`")
    add(f"- pinned branch: {json.dumps(design['pinned_branch_model'])}")
    for name, rule in design["rules"].items():
        add(f"    - {name}: {rule}")
    add(f"- verdict rule: {json.dumps(design['verdict_rule'])}")
    add(
        f"- synthetic reader validation: {len(synthetic['cases'])} cases, all hold {synthetic['all_hold']}; the "
        f"shared-node case reports segment indices "
        f"{synthetic['cases']['continuity_across_a_shared_grid_node']['segment_indices']}"
    )
    add(
        f"- reconstruction bias measured on the synthetic model at eps/dx = {bias_case['diffuseness_eps_over_dx']}: "
        f"max |bias| {_fmt(bias_case['max_abs_bias_over_dx'])} dx, variation over the sweep "
        f"{_fmt(bias_case['bias_variation_over_the_sweep_over_dx'])} dx"
    )
    add(
        f"- readout versus the frozen reconstruction on the real t = 0 state: sampler max difference "
        f"{_fmt(frozen_equivalence['sampler_max_abs_difference'])}; pinned column "
        f"{frozen_equivalence['pinned_branch']['column_index']} root "
        f"{_fmt(frozen_equivalence['pinned_branch']['t0_root_over_dx'])} dx on the frozen segment "
        f"{frozen_equivalence['pinned_branch']['t0_root_segment_index']}; checks "
        f"{json.dumps(frozen_equivalence['checks'])}"
    )
    add("")
    add("## C. Re-audit of the L1A-2x comparison error (the old evidence is untouched)")
    add(f"- raw rule margins (dx): {json.dumps({k: round(v, 8) for k, v in reaudit['raw_margins_dx'].items()})}")
    add(f"- raw margin spread: {_fmt(reaudit['raw_margin_spread_dx'])} dx")
    add(
        "- the spread is a genuine dimensional disagreement, reported in three separate pieces: "
        + json.dumps({k: _fmt(v) for k, v in reaudit["what_the_spread_is_made_of"].items() if v is not None})
    )
    add(
        f"- what L1A-2x recorded instead: a margin spread of "
        f"{_fmt(reaudit['how_l1a2x_reported_it']['recorded_margin_spread_across_rules_dx'])} dx, that is the "
        "spread of each rule's margin divided by that rule's own full displacement in primary units"
    )
    add(f"- criticism: {reaudit['how_l1a2x_reported_it']['criticism']}")
    add(f"- arithmetic note: {reaudit['how_l1a2x_reported_it']['arithmetic_note']}")
    add(
        "- L1A-2x measured reconstruction residuals (dx): "
        + json.dumps(
            {
                key: round(value, 10)
                for key, value in reaudit["l1a2x_measured_reconstruction_residuals_dx"].items()
                if isinstance(value, (int, float))
            }
        )
    )
    add("")
    add("## D. Verification of the sealed prerequisite evidence")
    for name, entry in prerequisites["stages"].items():
        add(
            f"- {name}: {entry['status']}, bound files {entry.get('bound_files')}, recorded verdict "
            f"`{entry.get('verdict')}`, manifest sha256 `{str(entry.get('manifest_sha256'))[:16]}...`, recorded "
            f"source hashes verified {entry.get('source_hashes_verified_from_the_worktree')} from the worktree "
            f"and {entry.get('source_hashes_verified_from_commit_history')} from the commit history"
        )
    add(f"- all verified: {prerequisites['all_verified']}; failures: {prerequisites['failures']}")
    add(
        f"- initial state: candidate `{identity['candidate_key']}` from `{identity['state_source']}`; identity "
        f"checks {json.dumps(identity['identity_checks'])}; parameter hash "
        f"`{identity['parameter_hash'][:16]}...`; geometry hash `{identity['geometry_hash'][:16]}...`"
    )
    add(
        f"- support cell {identity['support']['cell']} at gap {_fmt(identity['support']['gap_over_dx'])} dx, "
        f"|grad phi| {_fmt(identity['support']['gradient_norm'])}, wall normal "
        f"{identity['support']['wall_normal_into_solid']}, probe point {identity['support']['point']}"
    )
    add("")
    add("## E. The window and its substep ledger")
    add(
        f"- {ledger['substeps_recorded']} substep rows, each carrying the advective increment, the explicit "
        "chemical-potential increment, the implicit CG increment, the complete update, the tracked branch "
        "readout on the cumulative and the own-base field, the mass ledger, the closed-face flux, the CG "
        "diagnostics and the fixed-support probe"
    )
    add(
        f"- contact guard: {json.dumps(ledger['contact_guard'])} (frozen threshold {CONTACT_GAP_THRESHOLD_DX} dx, "
        "used only as a stop guard and never scored)"
    )
    add(f"- the support change that stopped L1A-2x: {json.dumps(ledger['segment_binding_switch'])}")
    add("")
    add("## F. Ingredient closure, conservation and the implicit solve")
    add(
        f"- mass: drift {_fmt(conservation['mass']['mass_drift_absolute'])} absolute / "
        f"{_fmt(conservation['mass']['mass_drift_relative'])} relative; worst substep drift "
        f"{_fmt(conservation['mass']['max_abs_substep_drift'])}; worst cell-wise recombination "
        f"{_fmt(conservation['recombination']['max_recombination_linf'])}"
    )
    add(f"- impermeable wall: worst closed-face flux {_fmt(conservation['wall']['max_abs_flux_on_closed_faces'])}")
    add(
        f"- implicit solve: {conservation['implicit_solve']['substeps']} solves, iterations "
        f"{conservation['implicit_solve']['min_iterations']}..{conservation['implicit_solve']['max_iterations']}, "
        f"worst relative residual {_fmt(conservation['implicit_solve']['max_relative_residual'])}, all converged "
        f"{conservation['implicit_solve']['all_converged']} against rtol {conservation['implicit_solve']['rtol']}"
    )
    add(
        f"- capture equivalence: worst |dphi| versus the eager public step "
        f"{_fmt(equivalence['max_phi_difference_vs_eager_public_step'])} and versus the jitted public step "
        f"{_fmt(equivalence['max_phi_difference_vs_jitted_public_step'])}; worst velocity difference "
        f"{_fmt(equivalence['max_velocity_difference_vs_jitted_public_step'])}; worst gap difference "
        f"{_fmt(equivalence['max_gap_difference_over_dx'])} dx"
    )
    add("")
    add("## G. Reconstruction-stable attribution")
    add(
        f"- cumulative displacement along the fixed wall normal (dx, positive toward the solid): complete "
        f"{_fmt(cumulative['full'])}, advective {_fmt(cumulative['adv'])}, Cahn-Hilliard update "
        f"{_fmt(cumulative['ch_pair'])} (explicit flux {_fmt(cumulative['ch_explicit'])} and implicit solve "
        f"{_fmt(cumulative['ch_implicit'])} halves of one update), the update measured directly as one field "
        f"{_fmt(cumulative['ch_pair_direct_over_dx'])}, independent interface-normal ray "
        f"{_fmt(cumulative['normal_ray_full_dx'])}"
    )
    add(
        f"- own-base view of the same increments: advective {_fmt(attribution['own_base_dn_dx']['adv'])}, update "
        f"{_fmt(attribution['own_base_dn_dx']['ch_pair'])}; measured walk-versus-cumulative noise "
        f"{_fmt(attribution['walk_noise_floor_dx'])} dx"
    )
    add(
        f"- arithmetic linearisation rules: pinned point {_fmt(attribution['linearised_dn_dx']['full'])}, "
        f"support cell {_fmt(attribution['linearised_cell_dn_dx']['full'])}"
    )
    add(
        "- per-macro-step table (dx; advective, CH explicit, CH implicit, CH update, complete, tracked root, "
        "binding segment):"
    )
    for row in attribution["per_macro"]:
        add(
            f"    - step {row['macro_step']:>2} t={_fmt(row['t'], 4)}: adv {_fmt(row['dn_adv'])}, ch_expl "
            f"{_fmt(row['dn_ch_explicit'])}, ch_impl {_fmt(row['dn_ch_implicit'])}, ch_update "
            f"{_fmt(row['dn_ch_pair'])}, complete {_fmt(row['dn_full'])}, root {_fmt(row['root_full_over_dx'])}, "
            f"segment {row['tracked_segment_index']}"
        )
    add("")
    add("## H. Static component reconstruction and the order decomposition")
    add(f"- method: {study['method']}")
    add(
        f"- the static forward reconstruction of the crossing is within "
        f"{_fmt(study['max_static_reconstruction_forward_over_dx'])} dx of the directly measured crossing and "
        f"the backward one within {_fmt(study['max_static_reconstruction_backward_over_dx'])} dx; the "
        f"operator recombination of the three ingredients against the complete increment is within "
        f"{_fmt(study['max_operator_recombination_over_dx'])} dx; the order symmetry is "
        f"{_fmt(study['max_order_symmetry_over_dx'])} dx and the symmetric half-increment nonlinearity is "
        f"{_fmt(study['max_half_increment_nonlinearity_over_dx'])} dx, against a tolerance of "
        f"{STATIC_RECONSTRUCTION_TOLERANCE_DX} dx; the reconstruction was resolved on "
        f"{study['macros_with_a_resolved_support_cell']} of {study['macros_total']} macro steps, and the "
        f"static gate holds: {study['holds']}"
    )
    add(
        f"- the level-based fraction of the earlier construction leaves the unit interval badly "
        f"(max |fraction| {_fmt(study['level_based_fraction_max_abs'])} over "
        f"{study['level_based_fraction_total']} cell fractions, "
        f"{study['level_based_fraction_inside_the_unit_interval']} of them inside [0, 1]), so the released "
        f"reconstruction never rescales an increment by it: {study['level_based_fraction_note']}"
    )
    add(f"- order-resolution attribution: {study['nonlinearity_attribution']}")
    add("")
    add("## I. Residual decomposition (raw dx, nothing normalised)")
    for name, value in residuals["terms_dx"].items():
        add(f"- {name}: {_fmt(value)}")
    add(f"- definition: {residuals['definition']}")
    add("")
    add("## J. Validity screen")
    add(f"- criteria: {json.dumps(validity['criteria'])}")
    add(f"- failures per guard: {json.dumps(validity['failures'])}")
    uniqueness = validity["macro_guards"]["frozen_global_contour_uniqueness"]
    add(
        f"- non-unique frozen global contour: {uniqueness['macros_non_unique']} of "
        f"{uniqueness['macros_total']} macro steps carry a support point that is equidistant from two contour "
        f"locations (statuses {json.dumps(uniqueness['statuses_seen'])}, margin "
        f"{_fmt(uniqueness['min_margin_over_dx'])}..{_fmt(uniqueness['max_margin_over_dx'])} dx, second "
        f"candidate at {json.dumps(uniqueness['second_candidate_distance_over_dx'])} dx); holds "
        f"{uniqueness['holds']} - {uniqueness['rule']}"
    )
    add(
        f"- closing quantities: attribution closure ok {validity['attribution_closure_ok']}, linearisation "
        f"closure ok {validity['linearisation_closure_ok']}, static reconstruction ok "
        f"{validity['static_reconstruction_ok']}, capture equivalence ok {validity['equivalence_ok']}"
    )
    add(
        f"- comparison error: {_fmt(validity['comparison_error_dx'])} dx "
        f"({json.dumps({k: _fmt(v) for k, v in validity['comparison_error_terms_dx'].items()})}); dominance "
        f"margin {_fmt(validity['dominance_margin_dx'])} dx against a floor of "
        f"{_fmt(validity['dominance_floor_dx'])} dx"
    )
    add(f"- unresolved: {validity['unresolved']}")
    add(
        f"- observed: maximum tracked step {_fmt(validity['observed']['max_step_over_dx'])} dx, maximum probe "
        f"offset {_fmt(validity['observed']['max_abs_probe_offset'])}, minimum gradient ratio to t = 0 "
        f"{_fmt(validity['observed']['min_gradient_ratio_to_t0'])}, maximum substep displacement over eps "
        f"{_fmt(validity['observed']['max_substep_displacement_over_eps'])}, binding segments "
        f"{validity['observed']['tracked_segment_indices']}, arc statuses {validity['observed']['arc_statuses']}"
    )
    add("")
    add("## K. Decision")
    add(f"- verdict `{decision['verdict']}`")
    add(f"- reason: {decision['reason']}")
    add(
        f"- rule margins (dx): {json.dumps({k: round(v, 8) for k, v in decision['rule_margins_dx'].items()})}; "
        f"comparison terms (dx): {json.dumps({k: round(v, 8) for k, v in decision['comparison_terms_dx'].items()})}; "
        f"margin over comparison error {_fmt(decision['margin_over_comparison_error'])}"
    )
    add(
        f"- the two Cahn-Hilliard halves are one update: cancellation ratio "
        f"{_fmt(decision['cahn_hilliard_cancellation_ratio'])}; they are reported as ingredients only and no "
        "mechanism is attached to either half separately"
    )
    add(f"- scope: {json.dumps(decision['scope'])}")
    add(f"- forbidden claims respected: {json.dumps(decision['forbidden_claims_respected'])}")
    if decision["verdict_withheld_because"]:
        add("- the verdict is withheld because:")
        for reason in decision["verdict_withheld_because"]:
            add(f"    - {reason}")
    add(
        f"- the displacement measured here is the pre-contact response of the frozen initial state over t <= "
        f"{_fmt(ledger['t_final'], 4)}; it is not a real impact and it is not extrapolated to the late contact time"
    )
    add("")
    add("## L. Cross-check of the identical window")
    add(
        f"- maximum difference to the sealed L1A-2x window over the steps that stage measured: gap "
        f"{_fmt(cross_check['max_gap_difference_over_dx'])} dx, level at the pinned point "
        f"{_fmt(cross_check['max_phi_at_point_difference'])}, tracked root "
        f"{_fmt(cross_check['max_root_difference_over_dx'])} dx"
    )
    add(
        f"- L1A-2x lost its pinned edge from macro step {cross_check['first_lost_macro_step']} (it could "
        f"measure its edge on {cross_check['macros_where_l1a2x_could_measure_its_pinned_edge']} of "
        f"{cross_check['macros_measured_by_each_stage']['l1a2x']} macro steps); this "
        f"stage reports that step as the support change {json.dumps(cross_check['support_change_this_stage'])}"
    )
    add(
        f"- final state hashes match the sealed window: {json.dumps(cross_check['final_state_hash_matches'])}; "
        f"macro steps measured by each stage: {json.dumps(cross_check['macros_measured_by_each_stage'])}"
    )
    add("")
    add("## M. Adversarial and falsification checks")
    add(
        "- the arithmetic rules rebuild the sealed L1A-2x arithmetic of the identical window: "
        + json.dumps(
            {
                key: {name: _fmt(value) for name, value in table.items()}
                for key, table in adversarial["checks"]["the_arithmetic_rules_reproduce_the_sealed_l1a2x_window"][
                    "max_differences_over_dx"
                ].items()
            }
        )
    )
    for name, value in adversarial["checks"].items():
        add(f"- `{name}`: {json.dumps(value, default=_json_default)}"[:900])
    add(f"- all checks hold: {adversarial['all_hold']}; failed checks: {adversarial['failed_checks']}")
    add("")
    add("## N. Quality, focused tests and hosted CI")
    add(f"- quality baseline sha256 `{quality['quality_baseline_sha256']}`")
    add(
        f"- frozen adjacent suite: {quality['frozen_adjacent_suite']['classification']}, suite failures "
        f"{json.dumps(quality['frozen_adjacent_suite']['suite_failures'])}"
    )
    for name in quality["frozen_adjacent_suite"]["failing"]:
        add(f"    - failing: {name}")
    add(
        f"- old lineage failure, reported separately: {quality['legacy_lineage_suite']['failing']} "
        f"({quality['legacy_lineage_suite']['classification']})"
    )
    add(f"- not run: {json.dumps(quality['not_run'])}")
    add(
        f"- declared new diagnostics {quality['declared_new_code_diagnostics']}, expectation "
        f"`{quality['ratchet_expectation']}`"
    )
    add("- hosted CI observed after push: `UNMEASURED` (a re-seal records the observed result)")
    add("")
    add("")
    add("## O. Post-window diagnosis of the failed gates (exploratory, no promotion rights)")
    add(f"- status: {diagnosis['status']}")
    add(f"- what failed: {diagnosis['what_failed']}")
    add(f"- why the prespecified primary rule fails: {diagnosis['why_the_primary_rule_fails']}")
    for name, value in diagnosis["numbers"].items():
        add(f"- {name}: {json.dumps(value, default=_json_default)}")
    add(f"- coherent families: {json.dumps(diagnosis['coherent_families'])}")
    add(f"- measured structure of the failure: {json.dumps(diagnosis['measured_structure_of_the_failure'])}")
    add(f"- reading caveats: {json.dumps(diagnosis['reading_caveats'])}")
    add(f"- consequence: {diagnosis['consequence']}")
    add("")
    add("## P. Exactly one next action")
    add(f"- {action['action']}")
    add(f"- bounded scope: {json.dumps(action['bounded_scope'])}")
    add(f"- must not be started in this stage: {action['must_not_be_started_in_this_stage']}")
    add("")
    report = "\n".join(lines) + "\n"
    payload = {
        "stage": STAGE,
        "verdict": decision["verdict"],
        "headline": (
            f"Over {ledger['macro_steps_run']} frozen macro steps (t <= {_fmt(ledger['t_final'], 4)}) of "
            f"{ledger['case_name']}, the reconstruction-stable branch readout moves the interface "
            f"{_fmt(cumulative['full'])} dx toward the solid: the advective increment contributes "
            f"{_fmt(cumulative['adv'])} dx and the Cahn-Hilliard update {_fmt(cumulative['ch_pair'])} dx, with a "
            f"raw dominance margin of {_fmt(decision['dominance_margin_dx'])} dx against a measured comparison "
            f"error of {_fmt(decision['comparison_error_dx'])} dx; the prespecified rule returns "
            f"`{decision['verdict']}`."
        ),
        "sections": [line for line in lines if line.startswith("## ")],
        "reference_times": [row["t"] for row in ledger["macro_rows"]],
        "readout_fingerprint_sha256": design["fingerprint_sha256"],
    }
    return payload, report


# ---------------------------------------------------------------------------
# Section H - evidence assembly, self-verification and the CI re-seal
# ---------------------------------------------------------------------------
def verify_evidence_manifest(out_dir: str | Path) -> dict[str, Any]:
    """Verify one assembled evidence directory against its own manifest, fail closed."""
    directory = Path(out_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise AuditBlocked(f"no manifest.json under {directory}")
    manifest = json.loads(manifest_path.read_text())
    problems: list[str] = []
    if manifest.get("stage") != STAGE:
        problems.append(f"stage mismatch: {manifest.get('stage')!r}")
    if manifest.get("baseline_head_sha") != BASELINE_HEAD_SHA:
        problems.append("baseline head sha mismatch")
    if manifest.get("verified_main_sha") != VERIFIED_MAIN_SHA:
        problems.append("verified main sha mismatch")
    if int(manifest.get("solver_contract_version", -1)) != 12:
        problems.append("the solver contract of the manifest is not 12")
    if manifest.get("verdict") not in DECISION_STATES:
        problems.append(f"verdict outside the prespecified vocabulary: {manifest.get('verdict')!r}")
    if int(manifest.get("macro_steps_run", 0)) > MAX_MACRO_STEPS:
        problems.append("the macro step budget was exceeded")
    for key, want in (
        ("trajectories_run", 1),
        ("control_arms_run", 0),
        ("contact_events_scored", 0),
        ("refinement_studies_run", 0),
        ("ml_training_runs", 0),
    ):
        if int(manifest.get(key, -1)) != want:
            problems.append(f"{key} must be {want}, found {manifest.get(key)!r}")
    for key in ("t8_recomputed", "l1b_data_generated", "solver_source_edited", "initializer_promoted"):
        if manifest.get(key) is not False:
            problems.append(f"{key} must be false")
    for key in ("no_promotion", "no_hosted_ci_claim_before_push"):
        if manifest.get(key) is not True:
            problems.append(f"{key} must be true")
    missing = [name for name in EVIDENCE_FILES if not (directory / name).is_file()]
    if missing:
        problems.append(f"missing required files: {sorted(missing)}")
    seal = stage_2x._recorded_evidence_seal(manifest)
    if not seal:
        problems.append("this stage's manifest declares no bound-file seal")
    for name, want in sorted(seal.items()):
        target = directory / name
        if not target.is_file():
            problems.append(f"missing bound file {name}")
        elif sha256_file(target) != want:
            problems.append(f"sha mismatch {name}")
    sources = stage_2x._verify_recorded_sources(manifest.get("source_hashes"))
    problems.extend(sources["problems"])
    if problems:
        raise AuditBlocked(f"evidence verification failed: {problems}")
    return {
        "stage": manifest.get("stage"),
        "stage_version": manifest.get("stage_version"),
        "profile": manifest.get("profile"),
        "evidence_class": manifest.get("evidence_class"),
        "verdict": manifest.get("verdict"),
        "baseline_head_sha": manifest.get("baseline_head_sha"),
        "verified_main_sha": manifest.get("verified_main_sha"),
        "readout_version": manifest.get("readout_version"),
        "readout_fingerprint_sha256": manifest.get("readout_fingerprint_sha256"),
        "bound_files": len(seal),
        "recorded_source_hashes": sources["checked"],
        "recorded_source_hashes_verified_from_the_worktree": len(sources["verified_from_the_worktree"]),
        "hosted_ci_reseal": manifest.get("hosted_ci_reseal"),
        "directory": str(directory),
        "problems": problems,
        "missing_required": missing,
        "verified": True,
    }


def assemble_evidence(result: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Write every evidence file, then hash the seal and verify the directory as a consumer would."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report_payload, report_text = render_report(result)
    for name in EVIDENCE_FILES:
        if name in ("l1a2y_final_report.json", "l1a2y_final_report.md", "manifest.json"):
            continue
        payload = result.get(name)
        if payload is None:
            raise AuditBlocked(f"no payload for the required evidence file {name}")
        (directory / name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    (directory / "l1a2y_final_report.json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    (directory / "l1a2y_final_report.md").write_text(report_text)
    ledger = result["substep_ledger.json"]
    decision = result["mechanism_decision.json"]
    provenance = result["source_and_provenance.json"]
    design = result["readout_design.json"]
    manifest = {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "profile": "early_window_reconstruction_stable",
        "evidence_class": "FROZEN_CHNS_BRANCH_ATTRIBUTION",
        "baseline_head_sha": BASELINE_HEAD_SHA,
        "verified_main_sha": VERIFIED_MAIN_SHA,
        "solver_contract_version": 12,
        "case_name": ledger["case_name"],
        "profile_N": ledger["profile_N"],
        "requested_dt": ledger["requested_dt"],
        "macro_steps_run": ledger["macro_steps_run"],
        "t_final": ledger["t_final"],
        "verdict": decision["verdict"],
        "trajectories_run": 1,
        "control_arms_run": 0,
        "contact_events_scored": 0,
        "refinement_studies_run": 0,
        "t8_recomputed": False,
        "l1b_data_generated": False,
        "ml_training_runs": 0,
        "solver_source_edited": False,
        "initializer_promoted": False,
        "no_promotion": True,
        "no_hosted_ci_claim_before_push": True,
        "readout_version": design["readout_version"],
        "readout_fingerprint_sha256": design["fingerprint_sha256"],
        "module_sha256": provenance["module_sha256"],
        "readout_module_sha256": provenance["readout_module_sha256"],
        "source_hashes": dict(provenance["frozen_source_sha256"]),
        "required_files": list(EVIDENCE_FILES),
        "files": {},
    }
    for name in EVIDENCE_FILES:
        if name == "manifest.json":
            continue
        manifest["files"][name] = sha256_file(directory / name)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    return verify_evidence_manifest(directory)


def reseal_ci_status(ci_status: str, head_sha: str, out_dir: str | Path) -> dict[str, Any]:
    """Record an observed post-push hosted CI result; rewrites metadata only, never physics."""
    if ci_status not in ("success", "failure", "partial", "NOT_RUN"):
        raise AuditBlocked(f"refusing to record an unverified CI status: {ci_status!r}")
    if len(head_sha) != 40 or any(character not in "0123456789abcdef" for character in head_sha):
        raise AuditBlocked(f"refusing to record an unverified head sha: {head_sha!r}")
    directory = Path(out_dir)
    quality_path = directory / "quality_and_blocker_matrix.json"
    if not quality_path.is_file():
        raise AuditBlocked(f"no quality_and_blocker_matrix.json under {directory}")
    quality = json.loads(quality_path.read_text())
    previous = (quality.get("hosted_ci") or {}).get("status_for_this_head", "UNMEASURED")
    quality["hosted_ci"] = {
        "status_for_this_head": ci_status,
        "observed_head_sha": head_sha,
        "previous_status": previous,
        "resealed_after_push": True,
        "rule": "no hosted result is claimed for a head that was not pushed and observed",
    }
    quality_path.write_text(json.dumps(quality, indent=2, sort_keys=True, default=_json_default) + "\n")
    report_path = directory / "l1a2y_final_report.md"
    report = report_path.read_text()
    line = f"- hosted CI observed after push: `{ci_status}` for head `{head_sha}` (previous record: `{previous}`)\n"
    marker = "- hosted CI observed after push"
    if marker in report:
        start = report.index(marker)
        end = report.index("\n", start) + 1
        report = report[:start] + line + report[end:]
    else:
        section = "\n## P. Exactly one next action"
        if section not in report:
            raise AuditBlocked("could not locate the report insertion point for the CI re-seal")
        report = report.replace(section, "\n" + line + section, 1)
    report_path.write_text(report)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["hosted_ci_reseal"] = {
        "performed": True,
        "status": ci_status,
        "observed_head_sha": head_sha,
        "physics_evidence_untouched": True,
    }
    files = manifest.get("files", {})
    for name in ("quality_and_blocker_matrix.json", "l1a2y_final_report.md"):
        if name in files:
            files[name] = sha256_file(directory / name)
    manifest["files"] = files
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n")
    return verify_evidence_manifest(directory)


# ---------------------------------------------------------------------------
# Section I - the driver
# ---------------------------------------------------------------------------
STAGES = (
    "preflight",
    "sealed",
    "identity",
    "design",
    "reader",
    "equivalence",
    "reaudit",
    "window",
    "static",
    "attribution",
    "closure",
    "validity",
    "decision",
    "adversarial",
    "quality",
    "assemble",
)

STAGE_DEPENDENCIES: dict[str, frozenset[str]] = {
    "preflight": frozenset(),
    "sealed": frozenset({"preflight"}),
    "identity": frozenset({"preflight", "sealed"}),
    "design": frozenset({"preflight", "sealed", "identity"}),
    "reader": frozenset({"preflight"}),
    "equivalence": frozenset({"preflight", "sealed", "identity"}),
    "reaudit": frozenset({"preflight"}),
    "window": frozenset({"preflight"}),
    "static": frozenset({"preflight", "window"}),
    "attribution": frozenset({"preflight", "window", "static"}),
    "closure": frozenset({"preflight", "window"}),
    "validity": frozenset({"attribution", "closure", "static"}),
    "decision": frozenset({"validity"}),
    "adversarial": frozenset({"decision", "reaudit", "equivalence"}),
    "quality": frozenset({"decision"}),
    "assemble": frozenset(set(STAGES) - {"assemble"}),
}


def plan_stages(stages: str = "all") -> list[str]:
    """Expand a requested stage list into the closure of its dependencies, in canonical order."""
    requested = [part.strip() for part in stages.split(",") if part.strip()]
    if not requested or "all" in requested:
        requested = list(STAGES)
    unknown = [part for part in requested if part not in STAGE_DEPENDENCIES]
    if unknown:
        raise AuditBlocked(f"unknown stages: {unknown}")
    needed: set[str] = set()
    pending = list(requested)
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        needed.add(name)
        pending.extend(sorted(STAGE_DEPENDENCIES[name]))
    return [name for name in STAGES if name in needed]


def run(*, out: str | Path | None = None, stages: str = "all", macro_steps: int = MAX_MACRO_STEPS) -> dict[str, Any]:
    """Run the requested stages (with their dependencies) and return every payload by evidence file name."""
    plan = plan_stages(stages)
    result: dict[str, Any] = {}
    started = time.perf_counter()
    progress(f"stage plan: {', '.join(plan)}")
    result["preflight.json"] = preflight_state()
    provenance = source_and_provenance(result["preflight.json"])
    result["source_and_provenance.json"] = provenance
    if "sealed" in plan:
        progress("re-hashing the sealed L1A-2t/u/v/w/x evidence and every recorded source hash")
        result["frozen_evidence_verification.json"] = verify_prerequisites()
    if "identity" in plan:
        progress("binding the frozen C2 initial state to the sealed identity")
        result["initial_state_identity.json"] = initial_state_identity()
    if "design" in plan:
        progress("pinning the tracked branch and freezing the readout design")
        result["readout_design.json"] = readout_design(result["initial_state_identity.json"])
    if "reader" in plan:
        progress("running the synthetic reader validation")
        result["synthetic_reader_validation.json"] = synthetic_reader_validation()
    if "equivalence" in plan:
        progress("checking the readout against the frozen reconstruction on the t = 0 state")
        result["frozen_reconstruction_equivalence.json"] = frozen_reconstruction_equivalence(
            result["initial_state_identity.json"]
        )
    if "reaudit" in plan:
        progress("re-auditing the L1A-2x comparison error from its sealed payload")
        result["comparison_error_reaudit.json"] = comparison_error_reaudit()
    if "window" in plan:
        progress("running the one sanctioned window with reconstruction-stable capture")
        ledger = run_window(macro_steps)
        result["substep_ledger.json"] = ledger
        progress(
            f"window done: {ledger['macro_steps_run']} macro steps, {ledger['substeps_recorded']} substeps, "
            f"t <= {_fmt(ledger['t_final'], 5)}, contact guard {ledger['contact_guard']['triggered']}"
        )
        result["trajectory_specification.json"] = {
            "stage": STAGE,
            "case_name": ledger["case_name"],
            "profile_N": ledger["profile_N"],
            "dt": ledger["requested_dt"],
            "substep_dt": ledger["parameters"]["substep_dt"],
            "macro_step_cap": MAX_MACRO_STEPS,
            "macro_steps_run": ledger["macro_steps_run"],
            "substeps_per_macro": ledger["substeps_per_macro"],
            "t_final": ledger["t_final"],
            "solver": "the frozen contract-12 phasefield.step path, re-executed by the frozen L1A-2x capture",
            "readout": "l1a2y_contour_branch_readout (reconstruction-stable contour branch)",
            "initial_state_hashes": ledger["initial_state"],
            "final_state_hashes": ledger["final_state"],
            "contact_guard": ledger["contact_guard"],
            "segment_binding_switch": ledger["segment_binding_switch"],
            "prohibitions": ledger["prohibitions"],
            "equivalence_note": ledger["equivalence_note"],
        }
    if "static" in plan:
        progress("static component reconstruction and order resolution at every macro boundary")
        result["static_reconstruction_study.json"] = static_reconstruction_study(result["substep_ledger.json"])
    if "attribution" in plan:
        progress("reconstruction-stable attribution and the residual decomposition")
        attribution = interface_attribution(result["substep_ledger.json"])
        result["interface_attribution.json"] = attribution
        if "static" not in plan:
            result["static_reconstruction_study.json"] = static_reconstruction_study(result["substep_ledger.json"])
        result["residual_decomposition.json"] = residual_decomposition(result["substep_ledger.json"], attribution)
    if "closure" in plan:
        progress("ingredient closure, conservation and the implicit solve")
        result["conservation_and_convergence.json"] = conservation_and_convergence(result["substep_ledger.json"])
        result["capture_equivalence.json"] = capture_equivalence_summary(result["substep_ledger.json"])
    if "validity" in plan:
        progress("validity screen")
        result["validity_screen.json"] = validity_screen(
            result["substep_ledger.json"],
            result["interface_attribution.json"],
            result["residual_decomposition.json"],
            result["static_reconstruction_study.json"],
            result["capture_equivalence.json"],
        )
    if "decision" in plan:
        result["mechanism_decision.json"] = decide(
            result["substep_ledger.json"],
            result["interface_attribution.json"],
            result["residual_decomposition.json"],
            result["static_reconstruction_study.json"],
            result["validity_screen.json"],
            result["capture_equivalence.json"],
        )
        result["next_action.json"] = next_action(result["mechanism_decision.json"])
        progress(f"decision: {result['mechanism_decision.json']['verdict']}")
    if "adversarial" in plan:
        progress("adversarial and falsification checks, including the identical-window cross-check")
        cross_check = l1a2x_window_cross_check(result["substep_ledger.json"])
        result["adversarial_and_falsification.json"] = adversarial_and_falsification(
            result["substep_ledger.json"],
            result["interface_attribution.json"],
            result["residual_decomposition.json"],
            result["static_reconstruction_study.json"],
            result["validity_screen.json"],
            result["mechanism_decision.json"],
            provenance,
            cross_check,
            result["comparison_error_reaudit.json"],
            result["capture_equivalence.json"],
        )
        if not result["adversarial_and_falsification.json"]["all_hold"]:
            progress(
                f"WARNING: adversarial checks failed: {result['adversarial_and_falsification.json']['failed_checks']}"
            )
    if "quality" in plan:
        result["quality_and_blocker_matrix.json"] = quality_and_blockers(result["mechanism_decision.json"]["verdict"])
    if "assemble" in plan:
        out_dir = _out_dir(out)
        result["manifest.json"] = assemble_evidence(result, out_dir)
        progress(f"evidence assembled and self-verified in {out_dir}")
    result["__elapsed_seconds__"] = time.perf_counter() - started
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default="all", help="comma separated stage names, or 'all'")
    parser.add_argument("--out", default=None, help="evidence directory (default: evidence/l1a2y)")
    parser.add_argument("--macro-steps", type=int, default=MAX_MACRO_STEPS)
    parser.add_argument("--verify-evidence", default=None, help="verify an assembled evidence directory")
    parser.add_argument("--reseal-ci-status", default=None)
    parser.add_argument("--reseal-head-sha", default=None)
    args = parser.parse_args(argv)
    if args.reseal_ci_status:
        if not args.reseal_head_sha:
            raise AuditBlocked("--reseal-ci-status requires --reseal-head-sha")
        report = reseal_ci_status(args.reseal_ci_status, args.reseal_head_sha, _out_dir(args.out))
        print(f"[{STAGE}] evidence resealed: {json.dumps(report, default=_json_default)}")
        return
    if args.verify_evidence:
        report = verify_evidence_manifest(args.verify_evidence)
        print(f"[{STAGE}] evidence verified: {json.dumps(report, default=_json_default)}")
        return
    try:
        result = run(out=args.out, stages=args.stages, macro_steps=args.macro_steps)
    except FROZEN_BLOCKED as blocked_error:
        raise SystemExit(f"[{STAGE}] blocked: {blocked_error}") from blocked_error
    decision = result.get("mechanism_decision.json", {})
    print(f"[{STAGE}] done in {result['__elapsed_seconds__']:.1f}s; verdict={decision.get('verdict')}")


if __name__ == "__main__":
    main()
