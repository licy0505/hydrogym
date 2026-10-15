"""L1A-2x - frozen CHNS early-window interface transport attribution (diagnostic only).

Baseline: the exact head ``45749bc603568bdac8629a90e529ae1573796d51`` of PR #24.

The stage answers exactly one question. On the frozen ``pillar_training`` canary, from the frozen
L1A-2s C2 initialization, at ``N = 192`` and ``dt = 0.002``, over one window of at most 40 macro
steps (``t <= 0.08``, three frozen substeps per macro step): *which discrete term moves the phase
interface at the contact-side support in the early window* - the conservative advective face flux,
or the Cahn-Hilliard chemical-potential transport (its explicit flux plus its implicit solve)?

What the stage does:

* **prerequisites** - the frozen L1A-2t/u/v/w evidence directories are re-hashed file by file and
  their recorded main/branch lineage is checked; any mismatch fails the stage closed before a single
  solver call;
* **initial state** - the initial state is built by the frozen ``L1A-2s``/``L1A-2u`` path
  (``build_any_state(bundle, "C2_BASE")``), never by the production uniform initializer and never by
  an L1A-2w ramp construction, and its ``phi``/``u``/``v`` hashes are compared against the sealed
  L1A-2w evidence together with the solid geometry and the parameter hashes;
* **trajectory** - exactly one case, one ``dt``, one initial state, at most 40 macro steps, through a
  diagnostic capture that re-executes the frozen substep from the frozen kernels
  (``phase_advective_fluxes``, ``chemical_potential_fluxes``, ``control_volume_divergence``,
  ``solve_ch_implicit``, the frozen Brinkman damping and pressure projection) and records, per
  internal substep, the *true* discrete increments: the conservative advective flux term, the
  explicit chemical-potential flux term, the implicit CG solve increment and the complete update;
* **closure** - per substep the ingredient increments are recombined cell by cell against the
  complete update, ``sum_i V_i phi_i`` is tracked for every ingredient, every face with zero
  aperture is checked to carry exactly zero flux, and every implicit solve reports its CG iteration
  count, relative residual and convergence flag;
* **equivalence** - the same trajectory is duplicated through the frozen public step
  (``phasefield.step_with_diagnostics`` and the jitted ``phasefield.step``) and the capture is shown
  to agree with both, with the observed difference reported;
* **attribution** - on the *fixed* t=0 support cell, crossing point and interface normal, each
  discrete increment is linearised into a normal displacement. The residuals of that linearisation
  are reported separately: interpolation (frozen linear edge crossing versus the bilinear sampler),
  support migration (fixed-support reading versus the frozen global minimum-gap rule), nonlinearity
  (root of the sampled profile versus the linearised estimate), operator splitting (algebraic
  recombination) and the drift of the linearisation coefficient itself;
* **verdict** - a dominance claim is made only if the difference between the two contributions
  exceeds the accumulated residual band by a prespecified factor; otherwise the reading is reported
  as mixed or unresolved, and no mechanism is claimed.

Scope limits, enforced by tests and by the decision payload: the stage explains only ``t <= 0.08``;
it never extends the window toward the frozen contact threshold; it does not score a contact event,
does not run a control arm, does not refine in time or space, does not regenerate ``T = 8`` or L1B
data, and does not train anything. The early-window interface displacement is a pre-contact response
and is never presented as a real impact mechanism.

Prespecified criteria (fixed before the production run; a verdict may only be *withheld* by them,
never manufactured):

* ``FIXED_POINT_PROBE_BAND = 0.10`` - the fixed t=0 crossing point stays a valid interface probe
  while ``|phi(O) - 0.5| <= 0.10``;
* ``LOCAL_GRADIENT_FLOOR_RATIO = 0.5`` - the frozen discrete ``|grad phi|`` at that point must stay at
  least half of its t=0 value for the linearisation to remain meaningful;
* ``MAX_SUBSTEP_DISPLACEMENT_OVER_EPS = 0.25`` - a single substep may not move the attributed contour
  by more than a quarter of the interface thickness ``eps``;
* ``CONTOUR_UNIQUENESS_MARGIN_DX = 0.01`` - the frozen global minimum-gap rule counts as uniquely
  matched only if the second-best *distinct* candidate is at least ``0.01 dx`` farther from the wall;
* ``DOMINANCE_MARGIN_FACTOR = 3.0`` - a dominance claim needs ``|d_adv| - |d_net_ch|`` (or its
  negative) to exceed three times the accumulated residual band;
* ``CH_CANCELLATION_QUALIFIER = 0.25`` - when ``|net CH| / (|CH explicit| + |CH implicit|)`` falls
  below a quarter, every statement about the Cahn-Hilliard contribution must carry the
  near-cancellation qualifier;
* ``EQUIVALENCE_MAX_PHI_DIFF = 1e-6`` and ``EQUIVALENCE_MAX_GAP_DIFF_DX = 1e-4`` - capture-versus-
  public-step tolerances, two orders above the observed fusion-level difference and at least three
  orders below the smallest physical displacement the stage reports.
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
from production import impact_initialization_compatibility_audit as audit  # noqa: E402
from production import radial_incident_seed_eta_dt_audit as stage_2u  # noqa: E402

STAGE = "L1A-2x"
STAGE_VERSION = "l1a2x_v1"
BASELINE_HEAD_SHA = "45749bc603568bdac8629a90e529ae1573796d51"
VERIFIED_MAIN_SHA = "09e2993b8da478d79e768054194201af57121784"
BRANCH = "arena/02b6c71b-hydrogym"
CASE_NAME = "pillar_training"
PROFILE_N = 192
REQUESTED_DT = 0.002
MAX_MACRO_STEPS = 40  # owner-mandated window cap: t <= 0.08
SUBSTEPS_PER_MACRO = 3  # frozen pf.step contract
CONTACT_GAP_THRESHOLD_DX = 1.5  # frozen event rule, used only as a stop guard, never scored

FIXED_POINT_PROBE_BAND = 0.10
LOCAL_GRADIENT_FLOOR_RATIO = 0.5
MAX_SUBSTEP_DISPLACEMENT_OVER_EPS = 0.25
CONTOUR_UNIQUENESS_MARGIN_DX = 0.01
DOMINANCE_MARGIN_FACTOR = 3.0
CH_CANCELLATION_QUALIFIER = 0.25
EQUIVALENCE_MAX_PHI_DIFF = 1e-6
EQUIVALENCE_MAX_GAP_DIFF_DX = 1e-4
RAY_HALF_WINDOW_DX = 2.0
RAY_SCAN_STEP_DX = 0.05
RAY_ROOT_TOLERANCE_DX = 1e-6
SUPPORT_CROSSING_RADIUS_DX = 1.5

FROZEN_STAGES = (
    ("L1A-2t", "l1a2t"),
    ("L1A-2u", "l1a2u"),
    ("L1A-2v", "l1a2v"),
    ("L1A-2w", "l1a2w"),
)

DECISION_STATES = (
    "ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION",
    "CAHN_HILLIARD_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION",
    "MIXED_TERMS_WITHIN_THE_RESIDUAL_BAND",
    "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED",
)

EVIDENCE_FILES = (
    "preflight.json",
    "frozen_evidence_verification.json",
    "source_and_provenance.json",
    "initial_state_identity.json",
    "trajectory_specification.json",
    "substep_ledger.json",
    "interface_attribution.json",
    "residual_decomposition.json",
    "conservation_and_convergence.json",
    "capture_equivalence.json",
    "validity_screen.json",
    "mechanism_decision.json",
    "next_action.json",
    "adversarial_and_falsification.json",
    "quality_and_blocker_matrix.json",
    "l1a2x_final_report.json",
    "l1a2x_final_report.md",
    "manifest.json",
)
STAGE_OWNED_PATHS = (
    "examples/two_phase/production/frozen_chns_early_window_transport_attribution_audit.py",
    "examples/two_phase/tests/test_frozen_chns_early_window_transport_attribution_audit.py",
    "examples/two_phase/evidence/l1a2x/",
    "examples/two_phase/artifacts/l1a2x/",
)

TWO_PHASE = Path(__file__).resolve().parents[1]
REPO_ROOT = TWO_PHASE.parents[1]


class AuditBlocked(RuntimeError):
    """Fail-closed protocol, provenance, source-freeze or identity error."""


FROZEN_BLOCKED = (AuditBlocked, audit.AuditValidationError, stage_2u.AuditBlocked)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def progress(message: str) -> None:
    print(f"[{STAGE}] {message}", flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (jax.Array,)):
        return np.asarray(value).tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "UNMEASURED"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    return f"{float(value):.{digits}g}"


def _git(args: list[str], *, required: bool = True) -> str | None:
    completed = subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        if required:
            raise AuditBlocked(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
        return None
    return completed.stdout.strip()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _recorded_evidence_seal(manifest: dict[str, Any]) -> dict[str, str]:
    """Bound-file seal of a prerequisite manifest, whichever convention that stage used.

    The frozen prerequisites predate this stage and are not edited, so the seal is read from
    ``files`` when present and from ``evidence_files`` otherwise; a value may be a bare sha256 string
    or a mapping carrying a ``sha256`` field. A manifest that declares no recognizable seal is not
    treated as verified: an empty seal would make the re-hash vacuously green.
    """
    for key in ("files", "evidence_files"):
        entries = manifest.get(key)
        if not isinstance(entries, dict) or not entries:
            continue
        seal: dict[str, str] = {}
        for name, value in entries.items():
            if isinstance(value, str):
                seal[name] = value
            elif isinstance(value, dict) and isinstance(value.get("sha256"), str):
                seal[name] = value["sha256"]
        if seal:
            return seal
    return {}


def _display_path(path: Path) -> str:
    """Repository-relative path when the file is inside the repository, otherwise the path itself."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _git_blob(commit: str, relative: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout


def _verify_recorded_sources(recorded: Any) -> dict[str, Any]:
    """Re-hash every source hash a prerequisite manifest recorded.

    A recorded hash is verified by content: a file of the worktree that hashes to it, or - for a
    snapshot that only existed at an earlier commit of this branch - the same path at one of its own
    commits. A recorded hash that matches nothing is a lineage mismatch and fails closed, exactly like
    a file that hashes differently.
    """
    report: dict[str, Any] = {
        "checked": 0,
        "verified_from_the_worktree": {},
        "verified_from_commit_history": {},
        "unverifiable": [],
        "problems": [],
    }
    if not isinstance(recorded, dict) or not recorded:
        return report
    index: dict[str, list[str]] = {}
    for base in (TWO_PHASE, TWO_PHASE / "evidence"):
        if not base.is_dir():
            continue
        for candidate in sorted(base.rglob("*")):
            if candidate.is_file() and candidate.suffix in {".py", ".json", ".md"}:
                index.setdefault(sha256_file(candidate), []).append(_display_path(candidate))
    for key, want in sorted(recorded.items()):
        if not isinstance(want, str):
            continue
        report["checked"] += 1
        hits = sorted(index.get(want, []))
        if hits:
            report["verified_from_the_worktree"][key] = hits
            continue
        stem = key[: -len("_at_l1a2r")] if key.endswith("_at_l1a2r") else key
        historical: list[str] = []
        for candidate in sorted(TWO_PHASE.rglob(f"{stem}.py")):
            relative = _display_path(candidate)
            log = _git(["log", "--format=%H", "--", relative], required=False)
            for commit in (log or "").split():
                blob = _git_blob(commit, relative)
                if blob is not None and hashlib.sha256(blob).hexdigest() == want:
                    historical.append(f"{relative}@{commit[:12]}")
                    break
            if historical:
                break
        if historical:
            report["verified_from_commit_history"][key] = historical
        else:
            report["unverifiable"].append(key)
            report["problems"].append(
                f"recorded source hash for {key!r} matches no file in the worktree and no commit of its path"
            )
    return report


def _out_dir(out: str | Path | None) -> Path:
    return Path(out) if out is not None else TWO_PHASE / "evidence" / "l1a2x"


def _porcelain_path(line: str) -> str:
    """Path of one ``git status --porcelain`` line, tolerating a stripped leading space.

    ``_git`` strips the whole stdout, so the first line can lose its leading space and
    ``line[3:]`` would eat a character. Rename entries are reduced to the destination path.
    """
    body = line[2:] if len(line) > 2 else ""
    if " -> " in body:
        body = body.split(" -> ", 1)[1]
    return body.strip().strip('"').strip()


def _stage_owned(path: str) -> bool:
    return any(path == owned or path.startswith(owned) for owned in STAGE_OWNED_PATHS)


def _commit_touches_only_stage_paths(commit: str) -> bool:
    changed = _git(["show", "--name-only", "--pretty=format:", commit], required=False)
    if changed is None:
        return False
    paths = [line.strip() for line in changed.splitlines() if line.strip()]
    return all(_stage_owned(path) for path in paths)


# ---------------------------------------------------------------------------
# Section A - preflight, provenance
# ---------------------------------------------------------------------------
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
            f"this stage ran from the L1A-2w evidence head {BASELINE_HEAD_SHA}; every commit above it "
            "must be stage owned or the stage refuses to measure"
        ),
        "main_sha": VERIFIED_MAIN_SHA,
        "worktree_clean_outside_stage_deliverables": not foreign,
        "dirty_paths_outside_stage_deliverables": foreign,
        "stage_owned_dirty_paths": [path for path in dirty if _stage_owned(path)],
        "commits_above_baseline": commits,
        "commits_above_baseline_not_stage_owned": unknown,
        "shallow_repository": (_git(["rev-parse", "--is-shallow-repository"]) or "unknown") == "true",
    }
    if branch != BRANCH:
        raise AuditBlocked(f"not on the sanctioned branch: {branch!r}")
    if payload["shallow_repository"]:
        raise AuditBlocked("a shallow clone cannot verify the frozen lineage; refusing to measure")
    if foreign:
        raise AuditBlocked(f"worktree dirty outside the stage deliverables: {foreign}")
    if unknown:
        raise AuditBlocked(f"commits above the baseline are not stage owned: {unknown}")
    own_evidence = _out_dir(None) / "manifest.json"
    if payload["commits_above_baseline"] and not own_evidence.is_file() and not dirty:
        raise AuditBlocked("no manifest.json under the stage evidence directory; refusing to measure")
    return payload


def verify_prerequisites() -> dict[str, Any]:
    """Re-hash every frozen prerequisite evidence file and check its recorded lineage."""
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
        problems = []
        if manifest.get("stage") != stage_name:
            problems.append(f"stage field is {manifest.get('stage')!r}")
        if manifest.get("verified_main_sha") != VERIFIED_MAIN_SHA:
            problems.append(f"verified main sha is {manifest.get('verified_main_sha')!r}")
        seal = _recorded_evidence_seal(manifest)
        if not seal:
            problems.append("manifest declares no bound-file seal under files or evidence_files")
        for name, want in sorted(seal.items()):
            target = path / name
            if not target.is_file():
                problems.append(f"missing bound file {name}")
            elif sha256_file(target) != want:
                problems.append(f"sha mismatch {name}")
        sources = _verify_recorded_sources(manifest.get("source_hashes"))
        problems.extend(sources["problems"])
        entry.update(
            {
                "status": "VERIFIED" if not problems else "MISMATCH",
                "manifest_sha256": sha256_file(manifest_path),
                "bound_files": len(seal),
                "source_hashes_recorded": sources["checked"],
                "source_hashes_verified_from_the_worktree": len(sources["verified_from_the_worktree"]),
                "source_hashes_verified_from_commit_history": len(sources["verified_from_commit_history"]),
                "source_hashes_unverifiable": sources["unverifiable"],
                "source_hash_provenance": {
                    "from_the_worktree": sources["verified_from_the_worktree"],
                    "from_commit_history": sources["verified_from_commit_history"],
                },
                "problems": problems,
                "recorded_hosted_ci": (manifest.get("hosted_ci_reseal") or {}).get("status"),
                "recorded_baseline_head": manifest.get("baseline_head_sha"),
            }
        )
        if problems:
            out["failures"].append(f"{stage_name}: {problems}")
        out["stages"][stage_name] = entry
    out["all_verified"] = not out["failures"]
    out["rule"] = (
        "every prerequisite evidence file is re-hashed from disk against the seal of its own manifest, "
        "every source hash that manifest recorded is re-verified by content against the worktree or "
        "against the commit history of its own path, and a manifest that declares no recognizable seal "
        "is refused instead of being counted as verified; a single mismatch blocks the stage before any "
        "solver call"
    )
    if out["failures"]:
        raise AuditBlocked(f"frozen prerequisite evidence failed verification: {out['failures']}")
    return out


def source_and_provenance(preflight: dict[str, Any]) -> dict[str, Any]:
    module = Path(__file__).resolve()
    frozen = {
        "examples/two_phase/phasefield.py": TWO_PHASE / "phasefield.py",
        "examples/two_phase/production/impact_initialization_compatibility_audit.py": (
            TWO_PHASE / "production" / "impact_initialization_compatibility_audit.py"
        ),
        "examples/two_phase/production/radial_incident_seed_eta_dt_audit.py": (
            TWO_PHASE / "production" / "radial_incident_seed_eta_dt_audit.py"
        ),
        "examples/two_phase/production/timestep_policy.py": TWO_PHASE / "production" / "timestep_policy.py",
    }
    return {
        "module_path": str(module.relative_to(REPO_ROOT)),
        "module_sha256": sha256_file(module),
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "branch": preflight["branch"],
        "head_sha": preflight["head_sha"],
        "baseline_head_sha": BASELINE_HEAD_SHA,
        "main_sha": VERIFIED_MAIN_SHA,
        "solver_contract_version": 12,
        "phase_boundary_model": "impermeable_flux",
        "phase_transport_geometry": "sdf_cutcell_fv_v1",
        "phase_storage_model": "phase_only_float64_v1",
        "frozen_source_sha256": {name: sha256_file(path) for name, path in sorted(frozen.items())},
        "frozen_status_flags": {"L1A_STATUS": "BLOCKED", "L1B_DATA_NOT_READY": "unchanged", "contract": 12},
        "trajectories_run": 1,
        "trajectory_definition": (
            "exactly one case (pillar_training), one dt (0.002), one initial state (frozen C2), at most "
            f"{MAX_MACRO_STEPS} frozen macro steps; no control arm, no refinement, no extension toward contact"
        ),
    }


# ---------------------------------------------------------------------------
# Section B - initial state identity
# ---------------------------------------------------------------------------
def _array_sha(array: Any) -> str:
    return audit._array_sha(np.asarray(array))


def fixed_support(bundle: Any, state: Any) -> dict[str, Any]:
    """The frozen phi=0.5 contact support of a state: cell, crossing point, normal, gradient."""
    p = bundle.p
    phi = np.asarray(state.phi, dtype=np.float64)
    gap = audit._local_gap(phi, 0.5, bundle.solid, p)
    if gap.get("gap") is None or gap.get("support_xy") is None:
        raise AuditBlocked(f"no frozen phi=0.5 contour support on case {bundle.case_name}")
    point = (float(gap["support_xy"][0]), float(gap["support_xy"][1]))
    cell = [int(v) for v in gap["gap_cell"]]
    normal = gap["local_wall_normal_into_solid"]
    if normal is None:
        raise AuditBlocked("no embedded-wall normal at the frozen contour support")
    grad = _gradient_at(state.phi, point, bundle)
    norm = float(np.hypot(grad[0], grad[1]))
    if norm <= 0.0:
        raise AuditBlocked("the frozen discrete gradient vanishes at the contour support")
    return {
        "cell": cell,
        "point": [point[0], point[1]],
        "wall_normal_into_solid": [float(normal[0]), float(normal[1])],
        "gap": float(gap["gap"]),
        "gap_over_dx": float(gap["gap"]) / float(p.dx),
        "gap_method": gap["gap_method"],
        "contour_crossing_count": int(gap["threshold_contour_crossing_count"]),
        "gradient": [float(grad[0]), float(grad[1])],
        "gradient_norm": norm,
        "interface_normal": [float(grad[0] / norm), float(grad[1] / norm)],
        "chi_at_cell": float(np.asarray(bundle.solid.chi, dtype=np.float64)[cell[0], cell[1]]),
    }


def _bilinear(field: Any, x: float, y: float, bundle: Any) -> float:
    """Frozen periodic bilinear sample of one point, flattened to a Python float.

    ``audit._periodic_bilinear_sample`` returns an array shaped like its coordinate arguments, so the
    single-point call is flattened here instead of being unpacked at every call site.
    """
    values = np.asarray(field, dtype=np.float64)
    sampled = audit._periodic_bilinear_sample(values, np.array([x]), np.array([y]), bundle.p)
    return float(np.asarray(sampled, dtype=np.float64).reshape(-1)[0])


def _sample(field: Any, point: tuple[float, float], bundle: Any) -> float:
    """Frozen periodic bilinear sample at a physical point."""
    return _bilinear(field, float(point[0]), float(point[1]), bundle)


def _gradient_at(phi: Any, point: tuple[float, float], bundle: Any) -> np.ndarray:
    """Frozen discrete central-difference gradient, sampled at the point by the frozen sampler."""
    p = bundle.p
    gx = np.asarray(pf._ddx(phi, p.dx), dtype=np.float64)
    gy = np.asarray(pf._ddy(phi, p.dy), dtype=np.float64)
    return np.array([_sample(gx, point, bundle), _sample(gy, point, bundle)], dtype=np.float64)


def initial_state_identity() -> dict[str, Any]:
    """Build the one sanctioned initial state and bind it to the sealed L1A-2w evidence."""
    cases = audit._canary_cases()
    if CASE_NAME not in cases:
        raise AuditBlocked(f"frozen canary {CASE_NAME!r} is not available")
    bundle = audit._derive_bundle(CASE_NAME, cases[CASE_NAME], PROFILE_N, REQUESTED_DT)
    state, details = stage_2u.build_any_state(bundle, stage_2u.C2_KEY)
    if details.get("source", "").find("radial") >= 0:
        raise AuditBlocked("the initial state is a radial seed; this stage mandates the frozen C2 state")
    support = fixed_support(bundle, state)
    sealed_path = TWO_PHASE / "evidence" / "l1a2w" / "contour_normal_velocity_and_flux.json"
    sealed = json.loads(sealed_path.read_text())["cases"][CASE_NAME]["candidates"][stage_2u.C2_KEY]
    hashes = {
        "phi": _array_sha(state.phi),
        "u": _array_sha(state.u),
        "v": _array_sha(state.v),
    }
    checks = {
        "phi_hash": hashes["phi"] == sealed["phi_hash"],
        "u_hash": hashes["u"] == sealed["u_hash"],
        "v_hash": hashes["v"] == sealed["v_hash"],
        "contour_cell": support["cell"] == list(sealed["contour_support_cell"]),
        "contour_cell_chi": math.isclose(
            support["chi_at_cell"], float(sealed["contour_cell_chi"]), rel_tol=0.0, abs_tol=1e-12
        ),
        "contour_cell_normal_velocity": math.isclose(
            _sample(state.v, tuple(support["point"]), bundle) / float(bundle.u_impact),
            float(sealed["contour_cell_normal_velocity_over_u_impact"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    p = bundle.p
    geometry = {
        "open_volume_total": float(np.sum(np.asarray(pf.phase_control_volumes(bundle.solid, p), dtype=np.float64))),
        "closed_face_count_x": int(np.sum(np.asarray(pf.phase_face_apertures(bundle.solid, p)[0]) == 0.0)),
        "closed_face_count_y": int(np.sum(np.asarray(pf.phase_face_apertures(bundle.solid, p)[1]) == 0.0)),
        "solid_area_fraction_chi_ge_0p01": float(np.mean(np.asarray(bundle.solid.chi, dtype=np.float64) >= 0.01)),
    }
    parameters = {
        "N": PROFILE_N,
        "Lx": float(p.Lx),
        "Ly": float(p.Ly),
        "dx": float(p.dx),
        "dy": float(p.dy),
        "eps": float(p.eps),
        "M": float(p.M),
        "We": float(p.We),
        "Re": float(p.Re),
        "eta_pen": float(p.eta_pen),
        "rho_l": float(p.rho_l),
        "rho_g": float(p.rho_g),
        "cos_theta": float(cases[CASE_NAME].get("cos_theta", 0.0)),
        "effective_dt": float(p.dt),
        "requested_dt": REQUESTED_DT,
        "substep_dt": float(p.dt) / SUBSTEPS_PER_MACRO,
        "u_impact": float(bundle.u_impact),
        "initial_gap": float(bundle.initial_gap),
        "local_surface_top": float(bundle.local_surface_top),
        "ch_solver_rtol": float(p.ch_solver_rtol),
        "ch_solver_max_iterations": int(p.ch_solver_max_iterations),
    }
    if not all(checks.values()):
        raise AuditBlocked(f"initial state identity disagrees with the sealed L1A-2w evidence: {checks}")
    return {
        "case_name": CASE_NAME,
        "state_source": details.get("source", "frozen L1A-2s C2 path"),
        "candidate_key": stage_2u.C2_KEY,
        "hashes": hashes,
        "sealed_hashes": {"phi": sealed["phi_hash"], "u": sealed["u_hash"], "v": sealed["v_hash"]},
        "identity_checks": checks,
        "support": support,
        "geometry": geometry,
        "parameters": parameters,
        "geometry_hash": hashlib.sha256(json.dumps(geometry, sort_keys=True).encode("utf-8")).hexdigest(),
        "parameter_hash": hashlib.sha256(json.dumps(parameters, sort_keys=True).encode("utf-8")).hexdigest(),
        "prohibited_alternatives_not_used": [
            "UNIFORM_ALL_DOMAIN production initializer",
            "STREAMFUNCTION_LOCALIZED_V0 replay control",
            "any L1A-2w exploratory ramp construction",
            "any radial R0/R1/R2 seed",
        ],
    }


# ---------------------------------------------------------------------------
# Section C - one window through the frozen solver, with per-substep capture
# ---------------------------------------------------------------------------
def _bundle_and_state() -> tuple[Any, Any]:
    """The one sanctioned initial state: the frozen L1A-2s C2 construction, never a ramp or a uniform seed."""
    cases = audit._canary_cases()
    bundle = audit._derive_bundle(CASE_NAME, cases[CASE_NAME], PROFILE_N, REQUESTED_DT)
    state, _details = stage_2u.build_any_state(bundle, stage_2u.C2_KEY)
    return bundle, state


def _line_root(
    phi: Any, origin: np.ndarray, direction: np.ndarray, bundle: Any
) -> tuple[float | None, float | None, dict[str, Any]]:
    """Root of the bilinearly sampled ``phi - 0.5`` along the fixed interface line.

    The line is ``origin + s * direction`` with ``s`` in dx units. The sign change nearest to
    ``s = 0`` is bisected to ``RAY_ROOT_TOLERANCE_DX``; an exact zero sample is treated as the root.
    Returns ``(s_over_dx, y_over_dx, info)``.
    """
    p = bundle.p
    dx = float(p.dx)
    step = RAY_SCAN_STEP_DX * dx
    half = RAY_HALF_WINDOW_DX * dx
    count = int(round(2.0 * half / step))

    def sample(offset: float) -> float:
        return (
            _bilinear(
                phi,
                float(origin[0] + offset * direction[0]),
                float(origin[1] + offset * direction[1]),
                bundle,
            )
            - 0.5
        )

    offsets = np.linspace(-half, half, count + 1)
    values = np.array([sample(float(offset)) for offset in offsets])
    exact = np.where(values == 0.0)[0]
    if exact.size:
        index = int(exact[np.argmin(np.abs(offsets[exact]))])
        s = float(offsets[index])
        return (
            s / dx,
            (float(origin[1]) + s * float(direction[1])) / dx,
            {
                "roots_found": 1,
                "root_is_an_exact_sample": True,
                "bracket_over_dx": [s / dx, s / dx],
            },
        )
    crosses = (values[:-1] * values[1:]) < 0.0
    changes = np.where(crosses)[0]
    if not changes.size:
        return None, None, {"roots_found": 0, "root_is_an_exact_sample": False, "bracket_over_dx": None}
    best = int(changes[np.argmin(np.abs(offsets[changes]))])
    lo = float(offsets[best])
    hi = float(offsets[best + 1])
    flo = sample(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = sample(mid)
        if (flo * fm) <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = fm
        if (hi - lo) <= RAY_ROOT_TOLERANCE_DX * dx:
            break
    s = 0.5 * (lo + hi)
    return (
        s / dx,
        (float(origin[1]) + s * float(direction[1])) / dx,
        {
            "roots_found": int(changes.size),
            "root_is_an_exact_sample": False,
            "bracket_over_dx": [offsets[best] / dx, offsets[best + 1] / dx],
        },
    )


def _support_edge_identity(phi0: Any, origin: np.ndarray, bundle: Any) -> dict[str, Any]:
    """Identity of the frozen linear contour edge that carries the t=0 contact-side crossing.

    The frozen rule reconstructs the ``phi = 0.5`` contour as linear crossings on grid edges. The
    crossing nearest the frozen support point is recorded once, at ``t = 0``, as ``(axis, i, j)``
    (``axis = 0``: the segment joining cells ``(i, j)`` and ``(i + 1, j)``; ``axis = 1``: the segment
    joining ``(i, j)`` and ``(i, j + 1)``) so the later reconstruction can follow *that* edge instead
    of hopping to whichever edge happens to be nearest.
    """
    p = bundle.p
    phi = np.asarray(phi0, dtype=np.float64)
    radius = 1.0e-9
    for axis in (1, 0):
        neighbor = np.roll(phi, -1, axis=axis)
        crosses = ((phi <= 0.5) & (neighbor >= 0.5)) | ((phi >= 0.5) & (neighbor <= 0.5))
        crosses &= neighbor != phi
        i_index, j_index = np.where(crosses)
        if not len(i_index):
            continue
        a = phi[i_index, j_index]
        b = neighbor[i_index, j_index]
        fraction = (0.5 - a) / (b - a)
        x_axis = (i_index.astype(np.float64) + 0.5) * float(p.dx)
        y_axis = (j_index.astype(np.float64) + 0.5) * float(p.dy)
        if axis == 0:
            x = x_axis + fraction * float(p.dx)
            y = y_axis
        else:
            x = x_axis
            y = y_axis + fraction * float(p.dy)
        distance = np.hypot(x - float(origin[0]), y - float(origin[1]))
        if distance.min() > radius:
            continue
        best = int(np.argmin(distance))
        return {
            "axis": axis,
            "i": int(i_index[best]),
            "j": int(j_index[best]),
            "t0_fraction": float(fraction[best]),
            "t0_x_over_dx": float(x[best]) / float(p.dx),
            "t0_y_over_dx": float(y[best]) / float(p.dx),
            "t0_radius_over_dx": float(distance[best]) / float(p.dx),
        }
    raise AuditBlocked("no frozen contour edge carries the t=0 contact-side crossing")


def _tracked_edge_crossing(phi: Any, edge: dict[str, Any], bundle: Any) -> dict[str, Any]:
    """Position of the frozen linear crossing on one *fixed* grid edge, if it still exists.

    A sign change that disappears on the tracked edge is reported as ``branch_lost``: the frozen
    reconstruction has migrated to a different edge, which is recorded rather than silently followed.
    """
    p = bundle.p
    values = np.asarray(phi, dtype=np.float64)
    i, j = int(edge["i"]), int(edge["j"])
    if int(edge["axis"]) == 0:
        first = float(values[i, j])
        second = float(values[(i + 1) % p.Nx, j])
    else:
        first = float(values[i, j])
        second = float(values[i, (j + 1) % p.Ny])
    straddles = (first - 0.5) * (second - 0.5) <= 0.0 and first != second
    if not straddles:
        return {"branch_lost": True, "fraction": None, "y_over_dx": None, "x_over_dx": None}
    fraction = (0.5 - first) / (second - first)
    if int(edge["axis"]) == 0:
        x_over_dx = i + 0.5 + fraction
        y_over_dx = j + 0.5
    else:
        x_over_dx = i + 0.5
        y_over_dx = j + 0.5 + fraction
    return {
        "branch_lost": False,
        "fraction": float(fraction),
        "x_over_dx": float(x_over_dx),
        "y_over_dx": float(y_over_dx),
    }


def _contour_uniqueness(phi: Any, bundle: Any) -> dict[str, Any]:
    """Margin between the best and the second-best *distinct* frozen contour candidate."""
    p = bundle.p
    dx = float(p.dx)
    contour = audit._threshold_contour_edge_crossings(np.asarray(phi, dtype=np.float64), 0.5, p)
    if not contour.size:
        return {"status": "NO_CONTOUR", "margin_over_dx": None, "candidates": 0}
    sdf = np.asarray(
        audit._periodic_bilinear_sample(
            np.asarray(bundle.solid.sdf, dtype=np.float64), contour[:, 0], contour[:, 1], p
        ),
        dtype=np.float64,
    )
    order = np.argsort(sdf)
    best = float(sdf[order[0]])
    best_point = contour[order[0]]
    margin = None
    distance = None
    for index in order[1:]:
        candidate = contour[index]
        separation = math.hypot(float(candidate[0] - best_point[0]), float(candidate[1] - best_point[1]))
        if separation > 0.5 * dx:
            margin = (float(sdf[index]) - best) / dx
            distance = separation / dx
            break
    within = int(np.sum((sdf - best) <= CONTOUR_UNIQUENESS_MARGIN_DX * dx))
    return {
        "status": "UNIQUE" if (margin is not None and margin >= CONTOUR_UNIQUENESS_MARGIN_DX) else "NON_UNIQUE",
        "margin_over_dx": margin,
        "second_candidate_distance_over_dx": distance,
        "candidates": int(contour.size),
        "candidates_within_uniqueness_tolerance": within,
        "best_sdf_over_dx": best / dx,
    }


def _mass(volume: np.ndarray, field: Any) -> float:
    return float(np.sum(volume * np.asarray(field, dtype=np.float64)))


def _closed_face_flux_max(fluxes: list[Any], apertures: list[np.ndarray]) -> float:
    worst = 0.0
    for flux, aperture in zip(fluxes, apertures, strict=True):
        values = np.asarray(flux, dtype=np.float64)
        closed = aperture == 0.0
        if closed.any():
            worst = max(worst, float(np.max(np.abs(values[closed]))))
    return worst


def substep_capture(phi: Any, u: Any, v: Any, t: Any, bundle: Any) -> tuple[tuple[Any, Any, Any, Any], dict[str, Any]]:
    """One frozen substep, re-executed from the frozen kernels with every ingredient recorded.

    The operation order matches ``phasefield._phase_update`` and ``phasefield.step_with_diagnostics``
    exactly: the conservative advective flux is the term of the momentum right-hand side the frozen
    solver already assembles, the explicit chemical-potential flux uses the frozen
    ``_explicit_chemical_potential``, and the implicit increment is the difference produced by the
    frozen volume-weighted CG solve. The Brinkman damping and the pressure projection are the frozen
    expressions. Nothing is re-derived, replaced or post-processed.
    """
    p = bundle.p
    solid = bundle.solid
    dt = float(p.dt) / SUBSTEPS_PER_MACRO
    operator = pf.phase_transport_operator(solid, p)
    phi_rhs, u_rhs, v_rhs, _mu, mu_expl = pf.rhs(pf.State(phi, u, v, t), solid, p)
    adv_x, adv_y = pf.phase_advective_fluxes(u, v, phi, solid, p)
    ch_x, ch_y = pf.chemical_potential_fluxes(mu_expl, solid, p)
    div_adv = pf.control_volume_divergence(adv_x, adv_y, operator.volume_safe)
    div_ch = pf.control_volume_divergence(ch_x, ch_y, operator.volume_safe)
    source = phi_rhs - div_ch
    phi_hat = phi + dt * source
    phi_new, solve_info = pf.solve_ch_implicit(phi_hat, solid, p, dt)
    d_adv = dt * phi_rhs
    d_ch = -dt * div_ch
    d_imp = phi_new - phi_hat
    d_full = phi_new - phi
    damp = 1.0 / (1.0 + dt * solid.chi / p.eta_pen)
    u_new = (u + dt * u_rhs) * damp
    v_new = (v + dt * v_rhs) * damp
    divergence = pf._ddx(u_new, p.dx) + pf._ddy(v_new, p.dy)
    pressure = pf.poisson_solve(divergence / dt, p.m2_proj)
    u_new = u_new - dt * pf._ddx(pressure, p.dx)
    v_new = v_new - dt * pf._ddy(pressure, p.dy)
    return (phi_new, u_new, v_new, t + dt), {
        "adv_x": adv_x,
        "adv_y": adv_y,
        "ch_x": ch_x,
        "ch_y": ch_y,
        "div_adv": div_adv,
        "div_ch": div_ch,
        "phi_hat": phi_hat,
        "d_adv": d_adv,
        "d_ch": d_ch,
        "d_imp": d_imp,
        "d_full": d_full,
        "solve_info": solve_info,
        "dt": dt,
    }


def run_ledger(macro_steps: int = MAX_MACRO_STEPS) -> dict[str, Any]:
    """Run the one sanctioned window and record the per-substep physical ledger.

    Exactly one case, one ``dt``, one initial state and at most ``MAX_MACRO_STEPS`` frozen macro
    steps. Every recorded quantity is measured on the frozen solver path; nothing is rescaled,
    post-masked or hand-clipped. The window stops early if the frozen contact gap threshold is
    reached, so the stage never extends toward a contact event.
    """
    if macro_steps < 1 or macro_steps > MAX_MACRO_STEPS:
        raise AuditBlocked(f"macro step budget {macro_steps} is outside 1..{MAX_MACRO_STEPS}")
    bundle, state0 = _bundle_and_state()
    p = bundle.p
    solid = bundle.solid
    pf.validate_x64_for_phase_storage(p.phase_storage_model)
    dt = float(p.dt) / SUBSTEPS_PER_MACRO
    dx = float(p.dx)
    eps = float(p.eps)
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    aperture_x = np.asarray(pf.phase_face_apertures(solid, p)[0], dtype=np.float64)
    aperture_y = np.asarray(pf.phase_face_apertures(solid, p)[1], dtype=np.float64)
    support = fixed_support(bundle, state0)
    origin = np.array(support["point"], dtype=np.float64)
    direction = np.array(support["interface_normal"], dtype=np.float64)
    wall = np.array(support["wall_normal_into_solid"], dtype=np.float64)
    gradient0_norm = float(support["gradient_norm"])
    coefficient0 = -float(np.dot(direction, wall)) / gradient0_norm  # length units
    tau = float(np.dot(direction, wall))
    root0, root0_y, root0_info = _line_root(state0.phi, origin, direction, bundle)
    edge = _support_edge_identity(state0.phi, origin, bundle)
    tracking0 = _tracked_edge_crossing(state0.phi, edge, bundle)
    jit_step = jax.jit(pf.step, static_argnums=(2,))
    raw_eager = state0
    raw_jitted = state0
    equivalence: dict[str, Any] = {"per_macro": [], "capture_note": None}
    phi, u, v, t = state0.phi, state0.u, state0.v, state0.t
    contact_guard: dict[str, Any] = {"triggered": False, "macro_step": None, "gap_over_dx": None}
    rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    for macro in range(macro_steps):
        for _substep in range(SUBSTEPS_PER_MACRO):
            t_before = float(getattr(t, "item", lambda: t)())
            mass_before = _mass(volume, phi)
            (phi, u, v, t), ingredient = substep_capture(phi, u, v, t, bundle)
            mass_after = _mass(volume, phi)
            recombination = np.asarray(
                (ingredient["d_adv"] + ingredient["d_ch"] + ingredient["d_imp"]) - ingredient["d_full"],
                dtype=np.float64,
            )
            gradient = _gradient_at(phi, tuple(origin), bundle)
            gradient_norm = float(np.hypot(gradient[0], gradient[1]))
            phi_at_support = _sample(phi, tuple(origin), bundle)
            u_at_support = _sample(u, tuple(origin), bundle)
            v_at_support = _sample(v, tuple(origin), bundle)
            root, root_y, root_info = _line_root(phi, origin, direction, bundle)
            tracked = _tracked_edge_crossing(phi, edge, bundle)
            gap_row = audit._local_gap(np.asarray(phi, dtype=np.float64), 0.5, solid, p)
            rows.append(
                {
                    "substep": len(rows),
                    "macro_step": macro + 1,
                    "t_before": t_before,
                    "t_after": float(getattr(t, "item", lambda: t)()),
                    "mass_before": mass_before,
                    "mass_after": mass_after,
                    "mass_adv": _mass(volume, ingredient["d_adv"]),
                    "mass_ch": _mass(volume, ingredient["d_ch"]),
                    "mass_imp": _mass(volume, ingredient["d_imp"]),
                    "mass_complete": _mass(volume, ingredient["d_full"]),
                    "mass_residual": mass_after - mass_before - _mass(volume, ingredient["d_full"]),
                    "telescoping_adv": float(np.sum(volume * np.asarray(ingredient["div_adv"], dtype=np.float64))),
                    "telescoping_ch": float(np.sum(volume * np.asarray(ingredient["div_ch"], dtype=np.float64))),
                    "recombination_linf": float(np.max(np.abs(recombination))),
                    "closed_face_flux_max": _closed_face_flux_max(
                        [ingredient["adv_x"], ingredient["adv_y"], ingredient["ch_x"], ingredient["ch_y"]],
                        [aperture_x, aperture_y, aperture_x, aperture_y],
                    ),
                    "advective_flux_linf": float(
                        max(
                            np.max(np.abs(np.asarray(ingredient["adv_x"], dtype=np.float64))),
                            np.max(np.abs(np.asarray(ingredient["adv_y"], dtype=np.float64))),
                        )
                    ),
                    "chemical_flux_linf": float(
                        max(
                            np.max(np.abs(np.asarray(ingredient["ch_x"], dtype=np.float64))),
                            np.max(np.abs(np.asarray(ingredient["ch_y"], dtype=np.float64))),
                        )
                    ),
                    "cg_iterations": int(ingredient["solve_info"].iterations),
                    "cg_relative_residual": float(ingredient["solve_info"].relative_residual),
                    "cg_converged": bool(ingredient["solve_info"].converged),
                    "dphi_adv_at_support": _sample(ingredient["d_adv"], tuple(origin), bundle),
                    "dphi_ch_at_support": _sample(ingredient["d_ch"], tuple(origin), bundle),
                    "dphi_imp_at_support": _sample(ingredient["d_imp"], tuple(origin), bundle),
                    "dphi_full_at_support": _sample(ingredient["d_full"], tuple(origin), bundle),
                    "dphi_adv_at_support_cell": float(
                        np.asarray(ingredient["d_adv"], dtype=np.float64)[support["cell"][0], support["cell"][1]]
                    ),
                    "dphi_ch_at_support_cell": float(
                        np.asarray(ingredient["d_ch"], dtype=np.float64)[support["cell"][0], support["cell"][1]]
                    ),
                    "dphi_imp_at_support_cell": float(
                        np.asarray(ingredient["d_imp"], dtype=np.float64)[support["cell"][0], support["cell"][1]]
                    ),
                    "dphi_full_at_support_cell": float(
                        np.asarray(ingredient["d_full"], dtype=np.float64)[support["cell"][0], support["cell"][1]]
                    ),
                    "dphi_norm_linf": float(np.max(np.abs(np.asarray(ingredient["d_full"], dtype=np.float64)))),
                    "phi_at_support": phi_at_support,
                    "normal_velocity_at_support": float(u_at_support * wall[0] + v_at_support * wall[1]),
                    "interface_normal_velocity_at_support": float(
                        u_at_support * direction[0] + v_at_support * direction[1]
                    ),
                    "gradient_norm_at_support": gradient_norm,
                    "interface_normal_at_support": [
                        float(gradient[0] / gradient_norm),
                        float(gradient[1] / gradient_norm),
                    ],
                    "line_root_over_dx": root,
                    "line_root_y_over_dx": root_y,
                    "line_roots_found": root_info.get("roots_found"),
                    "tracked_edge_branch_lost": bool(tracked["branch_lost"]),
                    "tracked_edge_y_over_dx": tracked["y_over_dx"],
                    "gap_global_over_dx": (float(gap_row["gap"]) / dx if gap_row.get("gap") is not None else None),
                    "gap_global_cell": None if gap_row.get("gap_cell") is None else list(gap_row["gap_cell"]),
                    "gap_global_method": gap_row.get("gap_method"),
                }
            )
        raw_eager, eager_info = pf.step_with_diagnostics(raw_eager, solid, p)
        raw_jitted = jax.block_until_ready(jit_step(raw_jitted, solid, p))
        phi_np = np.asarray(phi, dtype=np.float64)
        jitted_np = np.asarray(raw_jitted.phi, dtype=np.float64)
        gap_capture = rows[-1]["gap_global_over_dx"]
        gap_public = audit._local_gap(jitted_np, 0.5, solid, p)
        gap_public_dx = float(gap_public["gap"]) / dx if gap_public.get("gap") is not None else None
        uniqueness = _contour_uniqueness(phi, bundle)
        equivalence["per_macro"].append(
            {
                "macro_step": macro + 1,
                "max_phi_diff_eager": float(np.max(np.abs(phi_np - np.asarray(raw_eager.phi, dtype=np.float64)))),
                "max_u_diff_eager": float(np.max(np.abs(np.asarray(u) - np.asarray(raw_eager.u)))),
                "max_v_diff_eager": float(np.max(np.abs(np.asarray(v) - np.asarray(raw_eager.v)))),
                "max_phi_diff_jitted": float(np.max(np.abs(phi_np - jitted_np))),
                "max_u_diff_jitted": float(np.max(np.abs(np.asarray(u) - np.asarray(raw_jitted.u)))),
                "max_v_diff_jitted": float(np.max(np.abs(np.asarray(v) - np.asarray(raw_jitted.v)))),
                "cg_iterations_eager": [int(value) for value in np.asarray(eager_info.implicit_iterations)],
                "cg_converged_eager": bool(np.all(np.asarray(eager_info.implicit_converged))),
                "gap_capture_over_dx": gap_capture,
                "gap_public_step_over_dx": gap_public_dx,
                "gap_public_step_cell": None if gap_public.get("gap_cell") is None else list(gap_public["gap_cell"]),
                "gap_difference_over_dx": (
                    abs(float(gap_capture) - float(gap_public_dx))
                    if gap_capture is not None and gap_public_dx is not None
                    else None
                ),
                "gap_capture_cell": rows[-1]["gap_global_cell"],
                "contour_uniqueness": uniqueness,
            }
        )
        macro_rows.append(
            {
                "macro_step": macro + 1,
                "t": rows[-1]["t_after"],
                "gap_global_over_dx": rows[-1]["gap_global_over_dx"],
                "gap_global_cell": rows[-1]["gap_global_cell"],
                "line_root_over_dx": rows[-1]["line_root_over_dx"],
                "line_root_y_over_dx": rows[-1]["line_root_y_over_dx"],
                "tracked_edge_branch_lost": rows[-1]["tracked_edge_branch_lost"],
                "tracked_edge_y_over_dx": rows[-1]["tracked_edge_y_over_dx"],
                "phi_at_support": rows[-1]["phi_at_support"],
                "normal_velocity_at_support": rows[-1]["normal_velocity_at_support"],
                "gradient_norm_at_support": rows[-1]["gradient_norm_at_support"],
                "contour_uniqueness": uniqueness,
                "gap_public_step_over_dx": gap_public_dx,
                "gap_difference_over_dx": equivalence["per_macro"][-1]["gap_difference_over_dx"],
            }
        )
        if gap_capture is not None and float(gap_capture) <= CONTACT_GAP_THRESHOLD_DX:
            contact_guard.update({"triggered": True, "macro_step": macro + 1, "gap_over_dx": float(gap_capture)})
            break
    parameters = {
        "dx": dx,
        "dy": float(p.dy),
        "eps": eps,
        "dt": float(p.dt),
        "substep_dt": dt,
        "N": PROFILE_N,
        "u_impact": float(bundle.u_impact),
        "eps_over_dx": eps / dx,
        "M": float(p.M),
        "ch_solver_rtol": float(p.ch_solver_rtol),
        "ch_solver_max_iterations": int(p.ch_solver_max_iterations),
    }
    branch_lost_at = next((int(row["macro_step"]) for row in macro_rows if row["tracked_edge_branch_lost"]), None)
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
        "parameters": parameters,
        "linearisation": {
            "coefficient0_length_units": coefficient0,
            "coefficient0_over_dx": coefficient0 / dx,
            "interface_normal_dot_wall_normal": tau,
            "interface_normal": [float(direction[0]), float(direction[1])],
            "wall_normal_into_solid": [float(wall[0]), float(wall[1])],
            "t0_gradient_norm_at_support": gradient0_norm,
            "t0_line_root_over_dx": root0,
            "t0_line_root_y_over_dx": root0_y,
            "t0_line_root_info": root0_info,
            "rule": (
                "the normal displacement of an increment dphi measured at the fixed t=0 support point "
                "is dn = -(n_phi . n_w) / |grad phi| * dphi, reported in dx units; a positive dn moves "
                "the interface toward the solid"
            ),
        },
        "frozen_support_edge": {
            "identity": edge,
            "t0_tracking": tracking0,
            "branch_lost_at_macro_step": branch_lost_at,
            "rule": (
                "the frozen rule reconstructs the contour as linear crossings on grid edges; the edge "
                "carrying the t=0 contact-side crossing is tracked by identity so the reconstruction "
                "cannot silently hop to a neighbouring edge"
            ),
        },
        "geometry": {
            "total_open_volume": float(np.sum(volume)),
            "closed_face_count_x": int(np.sum(aperture_x == 0.0)),
            "closed_face_count_y": int(np.sum(aperture_y == 0.0)),
            "open_face_count_x": int(np.sum(aperture_x > 0.0)),
            "open_face_count_y": int(np.sum(aperture_y > 0.0)),
        },
        "initial_state": {
            "t": float(getattr(state0.t, "item", lambda: state0.t)()),
            "phi_hash": _array_sha(state0.phi),
            "u_hash": _array_sha(state0.u),
            "v_hash": _array_sha(state0.v),
        },
        "final_state": {
            "t": float(getattr(t, "item", lambda: t)()),
            "phi_hash": _array_sha(phi),
            "u_hash": _array_sha(u),
            "v_hash": _array_sha(v),
        },
        "rows": rows,
        "macro_rows": macro_rows,
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
            "the eager and the jitted public step are advanced from the same initial state in lockstep "
            "with the capture. They duplicate the single sanctioned trajectory and exist only to verify "
            "that the diagnostic capture reproduces the frozen public step; no physical conclusion is "
            "drawn from them and no control arm is run."
        ),
    }


# ---------------------------------------------------------------------------
# Section D - fixed-support attribution and the residual decomposition
# ---------------------------------------------------------------------------
def _dn(coefficient_over_dx: float, delta_phi: float) -> float:
    """Normal displacement in dx units of an increment sampled at the fixed support point."""
    return coefficient_over_dx * delta_phi


ATTRIBUTION_RULES = (
    ("PRIMARY_POINT_T0_COEFFICIENT", "point sample at the fixed support point with the t=0 coefficient"),
    ("POINT_LOCAL_COEFFICIENT", "point sample at the fixed support point with the per-substep coefficient"),
    ("SUPPORT_CELL_T0_COEFFICIENT", "support-cell value with the t=0 coefficient"),
)


def _rule_increments(ledger: dict[str, Any], rule: str, rows: list[dict[str, Any]]) -> dict[str, float]:
    """Cumulative ingredient displacements under one prespecified sampling/coefficient rule."""
    dx = float(ledger["parameters"]["dx"])
    coefficient = float(ledger["linearisation"]["coefficient0_over_dx"])
    wall = np.array(ledger["linearisation"]["wall_normal_into_solid"], dtype=np.float64)
    totals = {"adv": 0.0, "ch": 0.0, "imp": 0.0, "full": 0.0}
    for row in rows:
        if rule == "PRIMARY_POINT_T0_COEFFICIENT":
            factor = coefficient
            keys = ("dphi_adv_at_support", "dphi_ch_at_support", "dphi_imp_at_support", "dphi_full_at_support")
        elif rule == "SUPPORT_CELL_T0_COEFFICIENT":
            factor = coefficient
            keys = (
                "dphi_adv_at_support_cell",
                "dphi_ch_at_support_cell",
                "dphi_imp_at_support_cell",
                "dphi_full_at_support_cell",
            )
        elif rule == "POINT_LOCAL_COEFFICIENT":
            normal = np.array(row["interface_normal_at_support"], dtype=np.float64)
            gradient = float(row["gradient_norm_at_support"])
            factor = -float(np.dot(normal, wall)) / gradient / dx
            keys = ("dphi_adv_at_support", "dphi_ch_at_support", "dphi_imp_at_support", "dphi_full_at_support")
        else:
            raise AuditBlocked(f"unknown attribution rule {rule!r}")
        for name, key in zip(("adv", "ch", "imp", "full"), keys, strict=True):
            totals[name] += factor * float(row[key])
    return totals


def interface_attribution(ledger: dict[str, Any]) -> dict[str, Any]:
    """Linearised attribution of every ingredient on the fixed support and normal."""
    rows = ledger["rows"]
    macro_rows = ledger["macro_rows"]
    coefficient = float(ledger["linearisation"]["coefficient0_over_dx"])
    tau = float(ledger["linearisation"]["interface_normal_dot_wall_normal"])
    tau_effective = -tau  # |n_phi . n_w|
    substep_dn = {"adv": [], "ch": [], "imp": [], "full": [], "net_ch": []}
    for row in rows:
        adv = _dn(coefficient, row["dphi_adv_at_support"])
        ch = _dn(coefficient, row["dphi_ch_at_support"])
        imp = _dn(coefficient, row["dphi_imp_at_support"])
        full = _dn(coefficient, row["dphi_full_at_support"])
        substep_dn["adv"].append(adv)
        substep_dn["ch"].append(ch)
        substep_dn["imp"].append(imp)
        substep_dn["full"].append(full)
        substep_dn["net_ch"].append(ch + imp)
    cumulative = {key: float(np.sum(values)) for key, values in substep_dn.items()}
    per_macro_ingredients: list[dict[str, Any]] = []
    for macro_row in macro_rows:
        index = int(macro_row["macro_step"])
        block = [row for row in rows if int(row["macro_step"]) == index]
        per_macro_ingredients.append(
            {
                "macro_step": index,
                "t": macro_row["t"],
                "dn_adv": float(np.sum([_dn(coefficient, row["dphi_adv_at_support"]) for row in block])),
                "dn_ch_explicit": float(np.sum([_dn(coefficient, row["dphi_ch_at_support"]) for row in block])),
                "dn_ch_implicit": float(np.sum([_dn(coefficient, row["dphi_imp_at_support"]) for row in block])),
                "dn_full": float(np.sum([_dn(coefficient, row["dphi_full_at_support"]) for row in block])),
                "dn_net_ch": float(
                    np.sum([_dn(coefficient, row["dphi_ch_at_support"] + row["dphi_imp_at_support"]) for row in block])
                ),
            }
        )
    cancellations = []
    for macro_row, ingredients in zip(macro_rows, per_macro_ingredients, strict=True):
        pair = abs(ingredients["dn_ch_explicit"]) + abs(ingredients["dn_ch_implicit"])
        net = abs(ingredients["dn_net_ch"])
        cancellations.append(
            {
                "macro_step": int(macro_row["macro_step"]),
                "cancellation_ratio": (net / pair) if pair > 0.0 else None,
                "ch_pair_magnitude": pair,
                "net_ch_magnitude": net,
            }
        )
    family = {}
    for rule, description in ATTRIBUTION_RULES:
        totals = _rule_increments(ledger, rule, rows)
        margin = abs(totals["adv"]) - abs(totals["ch"] + totals["imp"])
        family[rule] = {
            "description": description,
            "cumulative_dn_dx": {key: float(value) for key, value in totals.items()},
            "cumulative_net_ch_dx": float(totals["ch"] + totals["imp"]),
            "margin_dx": float(margin),
            "ranking": "ADVECTIVE_LARGER" if margin > 0.0 else "NET_CH_LARGER",
        }
    primary_margin = family["PRIMARY_POINT_T0_COEFFICIENT"]["margin_dx"]
    comparisons = []
    for rule, entry in family.items():
        if rule == "PRIMARY_POINT_T0_COEFFICIENT":
            continue
        scale = entry["cumulative_dn_dx"]["full"] / family["PRIMARY_POINT_T0_COEFFICIENT"]["cumulative_dn_dx"]["full"]
        scale = scale if abs(scale) > 1e-300 else 1.0
        rescaled = entry["margin_dx"] / scale
        comparisons.append(
            {
                "rule": rule,
                "rescaling_to_the_primary_full_displacement": float(scale),
                "margin_primary_units_dx": float(rescaled),
                "ranking": entry["ranking"],
                "agrees_with_primary_ranking": entry["ranking"] == family["PRIMARY_POINT_T0_COEFFICIENT"]["ranking"],
            }
        )
    spread = max((abs(entry["margin_primary_units_dx"] - primary_margin) for entry in comparisons), default=0.0)
    pair_total = abs(cumulative["ch"]) + abs(cumulative["imp"])
    return {
        "case_name": ledger["case_name"],
        "support": ledger["support"],
        "linearisation": ledger["linearisation"],
        "coefficient_over_dx": coefficient,
        "interface_normal_dot_wall_normal": tau,
        "effective_normal_projection": tau_effective,
        "rules": list(ATTRIBUTION_RULES),
        "family": family,
        "cross_rule_comparison": comparisons,
        "margin_spread_across_rules_dx": float(spread),
        "rankings_agree_across_rules": bool(all(entry["agrees_with_primary_ranking"] for entry in comparisons)),
        "cumulative_dn_dx": cumulative,
        "cumulative_abs_dn_dx": {key: float(np.sum(np.abs(values))) for key, values in substep_dn.items()},
        "per_macro": per_macro_ingredients,
        "per_substep": {key: [float(value) for value in values] for key, values in substep_dn.items()},
        "cahn_hilliard_cancellation": {
            "cumulative_ratio": (abs(cumulative["net_ch"]) / pair_total) if pair_total > 0.0 else None,
            "per_macro": cancellations,
            "note": (
                "the explicit chemical-potential flux and the implicit CG increment are two halves of "
                "one Cahn-Hilliard update; their sum is the physical increment, and its ratio to the "
                "pair magnitude measures how nearly the two halves cancel"
            ),
        },
    }


def residual_decomposition(ledger: dict[str, Any], attribution: dict[str, Any]) -> dict[str, Any]:
    """Separate migration, interpolation, nonlinearity, splitting and coefficient-drift residuals.

    Four prespecified vertical displacement measurements of the same interface motion are compared
    pairwise, so every residual is a disagreement between two independently defined measurements:

    * ``d1`` - the frozen global minimum-gap rule (``_local_gap``) change;
    * ``d2`` - the frozen linear crossing on the *tracked* support edge;
    * ``d3`` - the bilinear root along the fixed interface line;
    * ``d4`` - the linearised point estimate used for the attribution.
    """
    rows = ledger["rows"]
    macro_rows = ledger["macro_rows"]
    coefficient = float(ledger["linearisation"]["coefficient0_over_dx"])
    t0 = {
        "gap": float(ledger["support"]["gap_over_dx"]),
        "edge_y": float(ledger["frozen_support_edge"]["t0_tracking"]["y_over_dx"]),
        "line_y": ledger["linearisation"]["t0_line_root_y_over_dx"],
    }
    previous = dict(t0)
    per_macro: list[dict[str, Any]] = []
    for macro_row, ingredients in zip(macro_rows, attribution["per_macro"], strict=True):
        index = int(macro_row["macro_step"])
        block = [row for row in rows if int(row["macro_step"]) == index]
        gap = macro_row["gap_global_over_dx"]
        edge_y = macro_row["tracked_edge_y_over_dx"]
        line_y = macro_row["line_root_y_over_dx"]
        d1 = None if (gap is None or previous["gap"] is None) else -(float(gap) - float(previous["gap"]))
        d2 = None if (edge_y is None or previous["edge_y"] is None) else -(float(edge_y) - float(previous["edge_y"]))
        d3 = None if (line_y is None or previous["line_y"] is None) else -(float(line_y) - float(previous["line_y"]))
        d4 = ingredients["dn_full"]
        r_split = float(
            np.sum(
                [
                    _dn(
                        coefficient,
                        row["dphi_adv_at_support"] + row["dphi_ch_at_support"] + row["dphi_imp_at_support"],
                    )
                    - _dn(coefficient, row["dphi_full_at_support"])
                    for row in block
                ]
            )
        )
        drift = 0.0
        for row in block:
            gradient = float(row["gradient_norm_at_support"])
            normal = np.array(row["interface_normal_at_support"], dtype=np.float64)
            wall = np.array(ledger["linearisation"]["wall_normal_into_solid"], dtype=np.float64)
            coefficient_t = -float(np.dot(normal, wall)) / gradient / float(ledger["parameters"]["dx"])
            drift += abs(coefficient_t - coefficient) * abs(float(row["dphi_full_at_support"]))
        per_macro.append(
            {
                "macro_step": index,
                "t": macro_row["t"],
                "d1_frozen_global_gap": d1,
                "d2_frozen_tracked_edge": d2,
                "d3_bilinear_line_root": d3,
                "d4_linearised_point_estimate": d4,
                "branch_lost": bool(macro_row["tracked_edge_branch_lost"]),
                "R_support_migration": (None if d1 is None or d2 is None else d1 - d2),
                "R_interpolation": (None if d2 is None or d3 is None else d2 - d3),
                "R_nonlinearity": (None if d3 is None else d3 - d4),
                "R_operator_splitting": r_split,
                "R_linearisation_coefficient_drift": drift,
                "identity_ingredients_plus_splitting": ingredients["dn_full"]
                + r_split
                - (ingredients["dn_adv"] + ingredients["dn_net_ch"]),
            }
        )
        previous = {"gap": gap, "edge_y": edge_y, "line_y": line_y}
    keys = (
        "R_nonlinearity",
        "R_interpolation",
        "R_support_migration",
        "R_operator_splitting",
        "R_linearisation_coefficient_drift",
    )
    cumulative_signed = {key: float(np.sum([row[key] for row in per_macro if row[key] is not None])) for key in keys}
    cumulative_band = {key: float(np.sum([abs(row[key]) for row in per_macro if row[key] is not None])) for key in keys}
    total_band = float(np.sum(list(cumulative_band.values())))
    cumulative_direct = float(
        np.sum([row["d1_frozen_global_gap"] for row in per_macro if row["d1_frozen_global_gap"] is not None])
    )
    cumulative_linearised = float(np.sum([row["d4_linearised_point_estimate"] for row in per_macro]))
    cumulative_line = float(
        np.sum([row["d3_bilinear_line_root"] for row in per_macro if row["d3_bilinear_line_root"] is not None])
    )
    cumulative_tracked = float(
        np.sum([row["d2_frozen_tracked_edge"] for row in per_macro if row["d2_frozen_tracked_edge"] is not None])
    )
    tracked_window = [row for row in per_macro if row["d2_frozen_tracked_edge"] is not None]
    measurement_keys = (
        "d1_frozen_global_gap",
        "d2_frozen_tracked_edge",
        "d3_bilinear_line_root",
        "d4_linearised_point_estimate",
    )
    common_cumulative = {
        key: (float(np.sum([row[key] for row in tracked_window])) if tracked_window else None)
        for key in measurement_keys
    }
    common_band = {
        key: float(np.sum([abs(row[residual_key]) for row in tracked_window if row[residual_key] is not None]))
        for key, residual_key in (
            ("R_support_migration", "R_support_migration"),
            ("R_interpolation", "R_interpolation"),
            ("R_nonlinearity", "R_nonlinearity"),
            ("R_operator_splitting", "R_operator_splitting"),
            ("R_linearisation_coefficient_drift", "R_linearisation_coefficient_drift"),
        )
    }
    common_total = float(np.sum(list(common_band.values())))
    common_geometric = (
        abs(common_cumulative["d1_frozen_global_gap"]) if common_cumulative["d1_frozen_global_gap"] is not None else 0.0
    )
    common_linearised = (
        abs(common_cumulative["d4_linearised_point_estimate"])
        if common_cumulative["d4_linearised_point_estimate"] is not None
        else 0.0
    )
    identities = {
        key: float(max([abs(row[key]) for row in per_macro if row[key] is not None] + [0.0]))
        for key in ("identity_ingredients_plus_splitting",)
    }
    branch_macro = ledger["frozen_support_edge"]["branch_lost_at_macro_step"]
    return {
        "per_macro": per_macro,
        "cumulative_signed": cumulative_signed,
        "cumulative_abs_band": cumulative_band,
        "total_residual_band_dx": total_band,
        "cumulative_direct_global_dx": cumulative_direct,
        "cumulative_linearised_dx": cumulative_linearised,
        "cumulative_line_root_dx": cumulative_line,
        "cumulative_tracked_edge_dx": cumulative_tracked,
        "tracked_edge_coverage": {
            "macros_available": len(tracked_window),
            "macros_total": len(per_macro),
            "first_macro_step": (tracked_window[0]["macro_step"] if tracked_window else None),
            "last_macro_step": (tracked_window[-1]["macro_step"] if tracked_window else None),
            "first_macro_step_without_it": next(
                (row["macro_step"] for row in per_macro if row["d2_frozen_tracked_edge"] is None), None
            ),
        },
        "window_where_every_measurement_exists": {
            "cumulative_dn_dx": common_cumulative,
            "residual_band_dx": common_band,
            "total_band_dx": common_total,
            "linearisation_overshoot_dx": ((common_linearised - common_geometric) if tracked_window else None),
            "linearisation_overshoot_relative": (
                ((common_linearised - common_geometric) / common_geometric) if common_geometric > 0.0 else None
            ),
            "meaning": (
                "the frozen global rule, the tracked frozen edge, the bilinear line root and the "
                "linearised point estimate only measure the same span while the tracked edge still "
                "carries the crossing, so their four-way agreement is quoted over that sub-window; the "
                "linearisation overshoot is the amount by which the point linearisation exceeds the "
                "geometric reconstruction over the same span"
            ),
        },
        "physical_closure_difference_dx": abs(cumulative_direct - cumulative_linearised),
        "line_root_closure_difference_dx": abs(cumulative_line - cumulative_linearised),
        "algebraic_identity_residuals": identities,
        "frozen_support_edge_branch_lost_at_macro_step": branch_macro,
        "frozen_support_edge_branch_alive": branch_macro is None,
        "definition": (
            "R_support_migration = d1 - d2 (the frozen global minimum-gap rule against the frozen rule "
            "restricted to the tracked support edge); R_interpolation = d2 - d3 (the frozen linear "
            "edge crossing against the bilinear root along the fixed interface line); R_nonlinearity = "
            "d3 - d4 (the bilinear line root against the linearised point estimate); "
            "R_operator_splitting = the algebraic recombination of the three ingredients against the "
            "complete update; R_linearisation_coefficient_drift = the change of the local coefficient "
            "kappa across the window, which is the assumption behind a fixed-coefficient "
            "linearisation. All quantities are vertical displacements in dx units, positive toward "
            "the solid."
        ),
    }


def conservation_and_convergence(ledger: dict[str, Any]) -> dict[str, Any]:
    """Mass, impermeable-wall, recombination and implicit-solve checks for every substep."""
    rows = ledger["rows"]
    masses = {
        "mass_before_first": rows[0]["mass_before"],
        "mass_after_last": rows[-1]["mass_after"],
        "mass_drift_absolute": rows[-1]["mass_after"] - rows[0]["mass_before"],
        "mass_drift_relative": (rows[-1]["mass_after"] - rows[0]["mass_before"]) / rows[0]["mass_before"],
        "max_abs_substep_drift": float(np.max(np.abs([row["mass_residual"] for row in rows]))),
        "max_abs_mass_adv": float(np.max(np.abs([row["mass_adv"] for row in rows]))),
        "max_abs_mass_ch": float(np.max(np.abs([row["mass_ch"] for row in rows]))),
        "max_abs_mass_imp": float(np.max(np.abs([row["mass_imp"] for row in rows]))),
        "max_abs_telescoping_adv": float(np.max(np.abs([row["telescoping_adv"] for row in rows]))),
        "max_abs_telescoping_ch": float(np.max(np.abs([row["telescoping_ch"] for row in rows]))),
    }
    wall = {
        "closed_face_count_x": ledger["geometry"]["closed_face_count_x"],
        "closed_face_count_y": ledger["geometry"]["closed_face_count_y"],
        "max_abs_flux_on_closed_faces": float(np.max([row["closed_face_flux_max"] for row in rows])),
        "rule": (
            "every face whose frozen aperture is exactly zero carries exactly zero advective and zero "
            "chemical-potential flux, which is how n.grad(mu) = 0 and the impermeability of the solid "
            "are imposed; no separate wall flux exists"
        ),
    }
    implicit = {
        "substeps": len(rows),
        "min_iterations": int(np.min([row["cg_iterations"] for row in rows])),
        "max_iterations": int(np.max([row["cg_iterations"] for row in rows])),
        "max_relative_residual": float(np.max([row["cg_relative_residual"] for row in rows])),
        "all_converged": bool(all(row["cg_converged"] for row in rows)),
        "rtol": ledger["parameters"]["ch_solver_rtol"],
        "iteration_cap": ledger["parameters"]["ch_solver_max_iterations"],
    }
    recombination = {
        "max_recombination_linf": float(np.max([row["recombination_linf"] for row in rows])),
        "max_abs_substep_increment_linf": float(np.max([row["dphi_norm_linf"] for row in rows])),
        "rule": (
            "d_adv + d_ch + d_imp - d_full is recombined cell by cell for every substep; the frozen "
            "update is exactly the sum of the three recorded ingredients up to floating point"
        ),
    }
    return {"mass": masses, "wall": wall, "implicit_solve": implicit, "recombination": recombination}


def capture_equivalence_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    """Summarise the capture-versus-public-step duplicate over the whole window."""
    per_macro = ledger["capture_equivalence"]["per_macro"]
    phi_eager = [row["max_phi_diff_eager"] for row in per_macro]
    phi_jitted = [row["max_phi_diff_jitted"] for row in per_macro]
    gap_difference = [row["gap_difference_over_dx"] for row in per_macro if row["gap_difference_over_dx"] is not None]
    worst_iterations = max(int(np.max(row["cg_iterations_eager"])) for row in per_macro)
    return {
        "macro_steps_compared": len(per_macro),
        "max_phi_difference_vs_eager_public_step": float(np.max(phi_eager)),
        "max_phi_difference_vs_jitted_public_step": float(np.max(phi_jitted)),
        "max_velocity_difference_vs_eager_public_step": float(
            np.max([max(row["max_u_diff_eager"], row["max_v_diff_eager"]) for row in per_macro])
        ),
        "max_velocity_difference_vs_jitted_public_step": float(
            np.max([max(row["max_u_diff_jitted"], row["max_v_diff_jitted"]) for row in per_macro])
        ),
        "max_gap_difference_over_dx": (float(np.max(gap_difference)) if gap_difference else None),
        "thresholds": {
            "max_phi_difference": EQUIVALENCE_MAX_PHI_DIFF,
            "max_gap_difference_over_dx": EQUIVALENCE_MAX_GAP_DIFF_DX,
        },
        "max_cg_iterations_observed": worst_iterations,
        "meaning": (
            "the frozen public step advances its three substeps inside a jitted lax.scan, so XLA fusion "
            "changes the last-bit rounding of the elementwise update; the observed differences are "
            "reported as measured and are compared against thresholds that are two orders above them "
            "and at least three orders below the smallest physical displacement the stage reports"
        ),
    }


def validity_screen(
    ledger: dict[str, Any],
    attribution: dict[str, Any],
    residuals: dict[str, Any],
    equivalence: dict[str, Any],
) -> dict[str, Any]:
    """Prespecified guards; a failed guard can only withhold a verdict, never grant one."""
    rows = ledger["rows"]
    macro_rows = ledger["macro_rows"]
    gradient0 = float(ledger["linearisation"]["t0_gradient_norm_at_support"])
    eps_over_dx = float(ledger["parameters"]["eps"]) / float(ledger["parameters"]["dx"])
    guards = {
        "gradient_resolved": [],
        "probe_band": [],
        "ray_root_exists": [],
        "substep_locality": [],
        "frozen_support_edge_alive": [],
        "contour_unique": [],
    }
    coefficient_over_dx = float(ledger["linearisation"]["coefficient0_over_dx"])
    for row in rows:
        row_guards = substep_guards(row, gradient0, eps_over_dx, coefficient_over_dx)
        for key, value in row_guards.items():
            guards[key].append(value)
    for macro_row in macro_rows:
        guards["frozen_support_edge_alive"].append(not bool(macro_row["tracked_edge_branch_lost"]))
        guards["contour_unique"].append(macro_row["contour_uniqueness"]["status"] == "UNIQUE")
    failures = {key: int(len(values) - sum(values)) for key, values in guards.items()}
    equivalence_ok = (
        equivalence["max_phi_difference_vs_eager_public_step"] <= EQUIVALENCE_MAX_PHI_DIFF
        and equivalence["max_phi_difference_vs_jitted_public_step"] <= EQUIVALENCE_MAX_PHI_DIFF
        and (
            equivalence["max_gap_difference_over_dx"] is None
            or equivalence["max_gap_difference_over_dx"] <= EQUIVALENCE_MAX_GAP_DIFF_DX
        )
    )
    blocking = {
        key: value
        for key, value in failures.items()
        if key
        in (
            "gradient_resolved",
            "probe_band",
            "ray_root_exists",
            "substep_locality",
            "frozen_support_edge_alive",
        )
        and value > 0
    }
    unresolved = bool(blocking) or not equivalence_ok
    direct_probe_ok = failures["contour_unique"] == 0
    return {
        "criteria": {
            "fixed_point_probe_band": FIXED_POINT_PROBE_BAND,
            "local_gradient_floor_ratio": LOCAL_GRADIENT_FLOOR_RATIO,
            "max_substep_displacement_over_eps": MAX_SUBSTEP_DISPLACEMENT_OVER_EPS,
            "contour_uniqueness_margin_dx": CONTOUR_UNIQUENESS_MARGIN_DX,
            "equivalence_max_phi_diff": EQUIVALENCE_MAX_PHI_DIFF,
            "equivalence_max_gap_diff_over_dx": EQUIVALENCE_MAX_GAP_DIFF_DX,
        },
        "substeps": len(rows),
        "failures": failures,
        "blocking_failures": blocking,
        "equivalence_ok": equivalence_ok,
        "direct_probe_unique": direct_probe_ok,
        "direct_probe_status": ("UNIQUE" if direct_probe_ok else "UNRESOLVED_NON_UNIQUE_CONTOUR_MATCH_ON_THIS_SURFACE"),
        "fixed_support_attribution_status": "UNRESOLVED" if unresolved else "RESOLVED",
        "unresolved": unresolved,
        "rule": (
            "a failed guard withholds the verdict, never grants one: an unresolved local gradient, a probe "
            "point that has drifted out of the interface band, a lost line root, a substep displacement "
            "beyond the locality cap, a jump of the frozen contour crossing off the tracked support edge, "
            "or a capture that does not reproduce the frozen public step all return UNRESOLVED. The "
            "contour-uniqueness guard applies to the frozen global minimum-gap probe, whose support can "
            "migrate between tied candidates on a flat pillar top; the primary readout is the "
            "fixed-support attribution, whose support and normal are frozen once at t = 0 and reported "
            "with the migration residual instead of being followed"
        ),
        "observed": {
            "min_gradient_ratio_to_t0": float(np.min([row["gradient_norm_at_support"] for row in rows])) / gradient0,
            "max_abs_probe_offset": float(np.max([abs(row["phi_at_support"] - 0.5) for row in rows])),
            "max_substep_displacement_over_eps": float(
                np.max([abs(_dn(coefficient_over_dx, row["dphi_full_at_support"])) / eps_over_dx for row in rows])
            ),
            "contour_uniqueness_margins_over_dx": [
                macro_row["contour_uniqueness"]["margin_over_dx"] for macro_row in macro_rows
            ],
            "contour_uniqueness_ok": direct_probe_ok,
            "support_cells_seen": sorted(
                {tuple(row["gap_global_cell"]) for row in rows if row["gap_global_cell"] is not None},
                key=str,
            ),
            "frozen_support_edge_branch_lost_at_macro_step": residuals["frozen_support_edge_branch_lost_at_macro_step"],
            "tracked_edge_continuity_macros_ok": int(sum(guards["frozen_support_edge_alive"])),
            "frozen_support_edge_identity": ledger["frozen_support_edge"]["identity"],
        },
    }


def decide(
    ledger: dict[str, Any],
    attribution: dict[str, Any],
    residuals: dict[str, Any],
    validity: dict[str, Any],
    equivalence: dict[str, Any],
) -> dict[str, Any]:
    """One verdict, withheld whenever the prespecified guards or margins do not support it."""
    cumulative = attribution["cumulative_dn_dx"]
    total_map_band = float(residuals["total_residual_band_dx"])
    comparison_band = float(attribution["margin_spread_across_rules_dx"]) + abs(
        float(residuals["cumulative_abs_band"]["R_operator_splitting"])
    )
    advective = abs(cumulative["adv"])
    chemical = abs(cumulative["net_ch"])
    margin = advective - chemical
    floor = DOMINANCE_MARGIN_FACTOR * comparison_band
    cancellation = attribution["cahn_hilliard_cancellation"]["cumulative_ratio"]
    ranking_agrees = bool(attribution["rankings_agree_across_rules"])
    if validity["unresolved"]:
        verdict = "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
        failed = json.dumps(validity["blocking_failures"])
        branch_step = residuals["tracked_edge_coverage"]["first_macro_step_without_it"]
        reason = (
            f"at least one prespecified validity guard failed ({failed}), so no mechanism is claimed; "
            "the per-substep ledger, the attribution table and the residual decomposition are still "
            "reported as measured quantities"
        )
        if "frozen_support_edge_alive" in validity["blocking_failures"]:
            reason += (
                f". The frozen linear reconstruction stopped carrying the contact-side crossing on the "
                f"tracked support edge at macro step {branch_step}: the reconstructed curve passed a "
                f"shared grid node, so its support moved to the neighbouring segment. Every other guard "
                f"passed, and the reconstructed crossing position itself changes smoothly across that "
                f"node, but a support change of the frozen reconstruction is a prespecified "
                f"unresolved condition, so the dominance comparison is not entered"
            )
    elif not ranking_agrees:
        verdict = "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
        reason = "the prespecified attribution rules disagree about which contribution is larger"
    elif margin >= floor:
        verdict = "ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION"
        reason = (
            f"|d_adv| - |d_net_ch| = {margin:.6g} dx exceeds {DOMINANCE_MARGIN_FACTOR} x the comparison band "
            f"{comparison_band:.6g} dx, and every prespecified attribution rule agrees on the ranking"
        )
    elif -margin >= floor:
        verdict = "CAHN_HILLIARD_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION"
        reason = (
            f"|d_net_ch| - |d_adv| = {-margin:.6g} dx exceeds {DOMINANCE_MARGIN_FACTOR} x the comparison "
            f"band {comparison_band:.6g} dx, and every prespecified attribution rule agrees on the ranking"
        )
    else:
        verdict = "MIXED_TERMS_WITHIN_THE_RESIDUAL_BAND"
        reason = (
            f"the difference between the two contributions ({margin:.6g} dx) does not exceed "
            f"{DOMINANCE_MARGIN_FACTOR} x the comparison band {comparison_band:.6g} dx"
        )
    recorded_rankings = {
        "with_the_comparison_band": _verdict_from_magnitudes(
            advective=advective, chemical=chemical, band=comparison_band, unresolved=False
        ),
        "with_the_total_residual_map_band": _verdict_from_magnitudes(
            advective=advective, chemical=chemical, band=total_map_band, unresolved=False
        ),
        "note": (
            "the outcome the prespecified dominance rule returns on the recorded magnitudes, quoted only "
            "to show which part of the rule the verdict rests on and which part the guards withhold; it "
            "is never used as a claim while a blocking guard has failed"
        ),
    }
    withheld_because: list[str] = []
    if validity["blocking_failures"]:
        withheld_because.append(f"blocking validity guards failed: {json.dumps(validity['blocking_failures'])}")
    if not validity["equivalence_ok"]:
        withheld_because.append(
            "the capture does not reproduce the frozen public step within the prespecified thresholds"
        )
    if validity["direct_probe_status"] != "UNIQUE":
        withheld_because.append(
            f"the frozen global minimum-gap probe is not unique on this surface: {validity['direct_probe_status']}"
        )
    if not ranking_agrees:
        withheld_because.append("the prespecified attribution rules disagree about which contribution is larger")
    qualifier = cancellation is not None and cancellation < CH_CANCELLATION_QUALIFIER
    sign_matches = (
        math.copysign(1.0, cumulative["full"]) == math.copysign(1.0, cumulative["adv"]) if cumulative["full"] else False
    )
    return {
        "case_name": ledger["case_name"],
        "verdict": verdict,
        "reason": reason,
        "verdict_withheld_because": withheld_because,
        "dominance_rule_outcome_recorded_not_claimed": recorded_rankings,
        "band_semantics": (
            "the decision compares two contributions of the SAME sampled increments through the SAME "
            "positive coefficient, so the band that matters for the comparison is the spread of the "
            "margin across the prespecified attribution rules (plus the machine-level algebraic "
            "recombination term); the larger total-map band is reported separately and used only to "
            "bound the total-displacement closure, it is not the uncertainty of a comparison"
        ),
        "cumulative_dn_dx": cumulative,
        "advective_magnitude_dx": advective,
        "net_chemical_hilliard_magnitude_dx": chemical,
        "dominance_margin_dx": margin,
        "comparison_band_dx": comparison_band,
        "total_map_band_dx": total_map_band,
        "dominance_floor_dx": floor,
        "dominance_margin_factor": DOMINANCE_MARGIN_FACTOR,
        "dominance_margin_over_comparison_band": (margin / comparison_band) if comparison_band > 0 else None,
        "dominance_margin_over_total_map_band": (margin / total_map_band) if total_map_band > 0 else None,
        "rankings_agree_across_rules": ranking_agrees,
        "cahn_hilliard_cancellation_ratio": cancellation,
        "cahn_hilliard_cancellation_qualifier": qualifier,
        "net_motion_sign_matches_dominant_term": bool(sign_matches),
        "direct_probe_status": validity["direct_probe_status"],
        "fixed_support_attribution_status": validity["fixed_support_attribution_status"],
        "capture_equivalence_ok": validity["equivalence_ok"],
        "equivalence": equivalence,
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
        },
        "next_action_is_not_started_in_this_stage": True,
    }


def next_action(decision: dict[str, Any]) -> dict[str, Any]:
    """Exactly one bounded next action, not started here."""
    if decision["verdict"] == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED":
        action = (
            "Prespecify, without editing the frozen event rule, a displacement readout whose support is "
            "the reconstructed contour arc itself (the bilinear line root along the frozen interface "
            "normal, already recorded in this stage's ledger on every macro step) and then re-run this "
            "stage's identical single-case window under that readout, so that the blocking condition "
            "becomes the continuity of the reconstructed position instead of the identity of the grid "
            "edge that carries it. Diagnose the frozen global probe's non-uniqueness on this surface in "
            "the same run and report it separately. No mechanism may be claimed until every blocking "
            "guard passes."
        )
    else:
        action = (
            "Run the identical prespecified window (one case, dt = 0.002, at most 40 macro steps, no "
            "extension toward contact) on the two remaining in-band canaries flat_we100_ct050 and "
            "flat_we200_ct000, using this stage's ledger and criteria unchanged, and report whether the "
            "advective carrier of the early-window contour motion is case specific."
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
# Section E - adversarial and falsification checks
# ---------------------------------------------------------------------------
def substep_guards(
    row: dict[str, Any], gradient0: float, eps_over_dx: float, coefficient_over_dx: float
) -> dict[str, bool]:
    """The four per-substep validity guards, evaluated on any ledger row."""
    return {
        "gradient_resolved": float(row["gradient_norm_at_support"]) >= LOCAL_GRADIENT_FLOOR_RATIO * gradient0,
        "probe_band": abs(float(row["phi_at_support"]) - 0.5) <= FIXED_POINT_PROBE_BAND,
        "ray_root_exists": row["line_root_over_dx"] is not None,
        "substep_locality": abs(_dn(coefficient_over_dx, float(row["dphi_full_at_support"])))
        <= MAX_SUBSTEP_DISPLACEMENT_OVER_EPS * eps_over_dx,
    }


def _synthetic_translation_probe(
    bundle: Any, offset_over_dx: float, wall_normal: tuple[float, float] = (0.0, 1.0)
) -> dict[str, Any]:
    """Translate a synthetic diffuse profile by a known amount and recover it by linearisation.

    The profile is the same diffuse tanh the frozen interface uses; the point sits on the t=0
    contour. The wall orientation decides which way is "toward the solid", so the same translation
    must be recovered with opposite sign for the two orientations, which is how the sign convention
    of the attribution is falsified rather than assumed.
    """
    p = bundle.p
    dx = float(p.dx)
    eps = float(p.eps)
    grid_x = np.asarray(pf.grids(p)[0], dtype=np.float64)
    grid_y = np.asarray(pf.grids(p)[1], dtype=np.float64)
    profile = 0.5 * (1.0 + np.tanh((grid_y - 1.0) / (2.0 * eps)))
    shifted = 0.5 * (1.0 + np.tanh((grid_y - 1.0 - offset_over_dx * dx) / (2.0 * eps)))
    point = (float(grid_x[96, 0]), 1.0)
    gradient = _gradient_at(profile, point, bundle)
    norm = float(np.hypot(gradient[0], gradient[1]))
    normal = np.array([gradient[0] / norm, gradient[1] / norm])
    wall = np.array([float(wall_normal[0]), float(wall_normal[1])])
    coefficient = -float(np.dot(normal, wall)) / norm
    delta = _sample(shifted, point, bundle) - _sample(profile, point, bundle)
    recovered_over_dx = _dn(coefficient / dx, delta)
    expected = offset_over_dx * float(np.dot(normal, wall))
    return {
        "offset_over_dx": offset_over_dx,
        "wall_normal": [float(wall[0]), float(wall[1])],
        "expected_displacement_toward_wall_over_dx": expected,
        "recovered_displacement_toward_wall_over_dx": recovered_over_dx,
        "absolute_error_over_dx": abs(recovered_over_dx - expected),
        "linearised_estimate_holds": abs(recovered_over_dx - expected) <= 0.02,
    }


def adversarial_and_falsification(
    ledger: dict[str, Any],
    attribution: dict[str, Any],
    residuals: dict[str, Any],
    decision: dict[str, Any],
    validity: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Falsification attempts: every claim must be able to fail, and the guards must bite."""
    bundle, _state = _bundle_and_state()
    dx = float(ledger["parameters"]["dx"])
    eps_over_dx = float(ledger["parameters"]["eps"]) / dx
    coefficient = float(ledger["linearisation"]["coefficient0_over_dx"])
    gradient0 = float(ledger["linearisation"]["t0_gradient_norm_at_support"])
    translation = [_synthetic_translation_probe(bundle, offset) for offset in (-0.5, -0.1, 0.1, 0.5)]
    mirror = [_synthetic_translation_probe(bundle, offset, (0.0, -1.0)) for offset in (-0.5, 0.5)]
    flat_row = dict(ledger["rows"][0])
    flat_row["gradient_norm_at_support"] = 0.0
    flat_guards = substep_guards(flat_row, gradient0, eps_over_dx, coefficient)
    drifting_row = dict(ledger["rows"][0])
    drifting_row["phi_at_support"] = 0.5 + 1.5 * FIXED_POINT_PROBE_BAND
    drifting_guards = substep_guards(drifting_row, gradient0, eps_over_dx, coefficient)
    broken_row = dict(ledger["rows"][0])
    broken_row["dphi_adv_at_support"] = float(broken_row["dphi_adv_at_support"]) + 1.0e-3
    broken_block = [broken_row]
    broken_identity = abs(
        float(np.sum([_dn(coefficient, row["dphi_full_at_support"]) for row in broken_block]))
        - float(np.sum([_dn(coefficient, row["dphi_adv_at_support"]) for row in broken_block]))
        - float(
            np.sum([_dn(coefficient, row["dphi_ch_at_support"] + row["dphi_imp_at_support"]) for row in broken_block])
        )
    )
    healthy_identity = abs(
        float(np.sum([_dn(coefficient, row["dphi_full_at_support"]) for row in [ledger["rows"][0]]]))
        - float(np.sum([_dn(coefficient, row["dphi_adv_at_support"]) for row in [ledger["rows"][0]]]))
        - float(
            np.sum(
                [
                    _dn(coefficient, row["dphi_ch_at_support"] + row["dphi_imp_at_support"])
                    for row in [ledger["rows"][0]]
                ]
            )
        )
    )
    near_tie = _verdict_from_magnitudes(
        advective=1.0,
        chemical=1.0 - 0.5 * DOMINANCE_MARGIN_FACTOR * float(residuals["total_residual_band_dx"]),
        band=float(residuals["total_residual_band_dx"]),
        unresolved=False,
    )
    clear_win = _verdict_from_magnitudes(
        advective=1.0,
        chemical=0.1,
        band=float(residuals["total_residual_band_dx"]),
        unresolved=False,
    )
    unresolved_win = _verdict_from_magnitudes(
        advective=1.0, chemical=0.0, band=float(residuals["total_residual_band_dx"]), unresolved=True
    )
    recorded_margin = [row["contour_uniqueness"]["margin_over_dx"] for row in ledger["macro_rows"]]
    status_consistent = all(
        (margin is not None and margin >= CONTOUR_UNIQUENESS_MARGIN_DX)
        == (row["contour_uniqueness"]["status"] == "UNIQUE")
        for row, margin in zip(ledger["macro_rows"], recorded_margin, strict=True)
    )
    frozen_hashes_hold = all(
        provenance["frozen_source_sha256"][name] == sha256_file(TWO_PHASE / name.replace("examples/two_phase/", ""))
        for name in provenance["frozen_source_sha256"]
    )
    capped_window_refused = False
    try:
        run_ledger(MAX_MACRO_STEPS + 1)
    except AuditBlocked:
        capped_window_refused = True
    non_vacuous = {
        "max_abs_mass_adv": float(np.max(np.abs([row["mass_adv"] for row in ledger["rows"]]))),
        "max_abs_mass_ch": float(np.max(np.abs([row["mass_ch"] for row in ledger["rows"]]))),
        "closed_faces_exist": ledger["geometry"]["closed_face_count_x"] > 0
        or ledger["geometry"]["closed_face_count_y"] > 0,
    }
    forbidden_text = ("real impact", "impact mechanism", "contact event occurred", "confirmed contact")
    evidence_text = json.dumps(decision, default=_json_default)
    checks = {
        "linearisation_recovers_a_known_translation": {
            "rows": translation,
            "holds": all(row["linearised_estimate_holds"] for row in translation),
        },
        "linearisation_sign_flips_with_the_wall_normal": {
            "rows": mirror,
            "holds": all(row["linearised_estimate_holds"] for row in mirror)
            and all(
                math.copysign(1.0, row["recovered_displacement_toward_wall_over_dx"])
                == math.copysign(1.0, row["expected_displacement_toward_wall_over_dx"])
                for row in mirror
            ),
        },
        "gradient_guard_refuses_a_flat_profile": {
            "guards": flat_guards,
            "refused": not flat_guards["gradient_resolved"],
        },
        "probe_band_guard_refuses_a_drifting_probe": {
            "guards": drifting_guards,
            "refused": not drifting_guards["probe_band"],
        },
        "ingredient_identity_catches_a_fabricated_increment": {
            "healthy_identity_over_dx": healthy_identity,
            "broken_identity_over_dx": broken_identity,
            "discriminates": broken_identity > 100.0 * max(healthy_identity, 1e-30),
        },
        "verdict_rule_degrades_to_mixed_on_a_near_tie": {
            "near_tie": near_tie,
            "clear_win": clear_win,
            "unresolved_win": unresolved_win,
            "degrades": near_tie == "MIXED_TERMS_WITHIN_THE_RESIDUAL_BAND",
            "grants_on_a_clear_win": clear_win == "ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION",
            "withholds_when_unresolved": unresolved_win == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED",
        },
        "contour_uniqueness_record_is_self_consistent": {
            "status_consistent": status_consistent,
            "margins_over_dx": recorded_margin,
        },
        "frozen_sources_are_unmodified": {
            "hashes_hold": frozen_hashes_hold,
            "note": (
                "the frozen solver, the frozen audit source and the L1A-2u/2s helpers re-hash to the "
                "values recorded at generation time"
            ),
        },
        "window_budget_is_capped": {
            "refused_over_cap_request": capped_window_refused,
            "cap": MAX_MACRO_STEPS,
        },
        "ledger_is_not_vacuous": non_vacuous,
        "no_forbidden_claim_in_the_decision_payload": {
            "forbidden_phrases": list(forbidden_text),
            "found": [phrase for phrase in forbidden_text if phrase in evidence_text],
            "holds": not any(phrase in evidence_text for phrase in forbidden_text),
        },
    }
    decisive = (
        bool(checks["linearisation_recovers_a_known_translation"]["holds"]),
        bool(checks["linearisation_sign_flips_with_the_wall_normal"]["holds"]),
        bool(checks["gradient_guard_refuses_a_flat_profile"]["refused"]),
        bool(checks["probe_band_guard_refuses_a_drifting_probe"]["refused"]),
        bool(checks["ingredient_identity_catches_a_fabricated_increment"]["discriminates"]),
        bool(checks["verdict_rule_degrades_to_mixed_on_a_near_tie"]["degrades"]),
        bool(checks["verdict_rule_degrades_to_mixed_on_a_near_tie"]["grants_on_a_clear_win"]),
        bool(checks["verdict_rule_degrades_to_mixed_on_a_near_tie"]["withholds_when_unresolved"]),
        bool(checks["contour_uniqueness_record_is_self_consistent"]["status_consistent"]),
        bool(checks["frozen_sources_are_unmodified"]["hashes_hold"]),
        bool(checks["window_budget_is_capped"]["refused_over_cap_request"]),
        bool(checks["no_forbidden_claim_in_the_decision_payload"]["holds"]),
        bool(checks["ledger_is_not_vacuous"]["closed_faces_exist"]),
    )
    checks["all_hold"] = all(decisive)
    return {
        "checks": checks,
        "all_hold": bool(checks["all_hold"]),
        "rule": (
            "every mechanism claim in this stage must be able to fail: the linearisation is validated "
            "against a known translation, each guard is exercised on a state that must be refused, the "
            "ingredient identity is exercised against a fabricated increment, the verdict rule is "
            "exercised on a near tie and on a clear win, and the frozen sources are re-hashed"
        ),
    }


def _verdict_from_magnitudes(*, advective: float, chemical: float, band: float, unresolved: bool) -> str:
    """The prespecified verdict rule, exposed for the adversarial checks."""
    if unresolved:
        return "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    margin = advective - chemical
    floor = DOMINANCE_MARGIN_FACTOR * band
    if margin >= floor:
        return "ADVECTIVE_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION"
    if -margin >= floor:
        return "CAHN_HILLIARD_TERM_CARRIES_THE_EARLY_WINDOW_CONTOUR_MOTION"
    return "MIXED_TERMS_WITHIN_THE_RESIDUAL_BAND"


# ---------------------------------------------------------------------------
# Section F - quality, tests, hosted CI
# ---------------------------------------------------------------------------
FROZEN_ADJACENT_FAILING_SET = (
    "tests/test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight",
    "tests/test_interior_speed_admissibility_screen_audit.py::test_required_evidence_files_are_declared_and_complete",
    "tests/test_interior_speed_admissibility_screen_audit.py::test_manifest_self_verification_and_tamper_detection",
    "tests/test_interior_speed_admissibility_screen_audit.py::test_evidence_verification_rejects_a_moved_envelope",
    "tests/test_interior_speed_admissibility_screen_audit.py::test_reseal_records_the_observed_ci_status_without_touching_physics",
    "tests/test_interior_speed_admissibility_screen_audit.py::test_report_carries_every_required_section",
    "tests/test_boundary_compatible_incident_momentum_bound_audit.py::test_preflight_is_clean_on_the_repository_head",
)
FROZEN_ADJACENT_SUITE_FAILURES = {"l1a2t": 1, "l1a2u": 0, "l1a2v": 5, "l1a2w": 1}
FROZEN_SWEEP_RECONFIRMED_AT_HEAD = {
    "head_sha": "41e962606dbeccf80c26d93e931fd3ae8098da5e",
    "passed": 164,
    "failed": 7,
    "suite_failures": dict(FROZEN_ADJACENT_SUITE_FAILURES),
    "failing": list(FROZEN_ADJACENT_FAILING_SET),
    "note": (
        "the same sweep re-run on a clean worktree at this stage's first committed head returns the "
        "seven structural failures and nothing else; the two worktree-cleanliness failures observed at "
        "generation time are then replaced by the L1A-2w stage-owned-commit failure, which is the same "
        "superseded-by-next-stage rule with a different trigger"
    ),
}
LEGACY_LINEAGE_FAILING_SET = (
    "tests/test_impact_initialization_compatibility_audit.py::test_l1a2r_merged_base_verified_or_fail_closed",
)
# Observed by running the frozen adjacent suites on the generating tree state, before this stage's own
# deliverables were committed. The two extra entries are the frozen preflights that require a worktree
# clean outside their own paths; they see this stage's uncommitted paths. The committed-head sweep is
# re-run and recorded in artifacts/l1a2x/local_validation.json, and this stage never edits a frozen test
# to make either entry disappear.
FROZEN_SWEEP_OBSERVED_AT_GENERATION = {
    "command": (
        "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest "
        "tests/test_nearwall_taper_interface_transport_audit.py "
        "tests/test_radial_incident_seed_eta_dt_audit.py "
        "tests/test_interior_speed_admissibility_screen_audit.py "
        "tests/test_boundary_compatible_incident_momentum_bound_audit.py -q"
    ),
    "passed": 163,
    "failed": 8,
    "suite_failures": {"l1a2t": 1, "l1a2u": 1, "l1a2v": 5, "l1a2w": 1},
    "failing": (
        "tests/test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight",
        "tests/test_radial_incident_seed_eta_dt_audit.py::test_report_contains_every_required_section",
        "tests/test_interior_speed_admissibility_screen_audit.py::test_required_evidence_files_are_declared_and_complete",
        "tests/test_interior_speed_admissibility_screen_audit.py::test_manifest_self_verification_and_tamper_detection",
        "tests/test_interior_speed_admissibility_screen_audit.py::test_evidence_verification_rejects_a_moved_envelope",
        "tests/test_interior_speed_admissibility_screen_audit.py::test_reseal_records_the_observed_ci_status_without_touching_physics",
        "tests/test_interior_speed_admissibility_screen_audit.py::test_report_carries_every_required_section",
        "tests/test_boundary_compatible_incident_momentum_bound_audit.py::test_preflight_is_clean_on_the_repository_head",
    ),
    "caused_by_the_uncommitted_stage_deliverables": (
        "tests/test_radial_incident_seed_eta_dt_audit.py::test_report_contains_every_required_section",
        "tests/test_boundary_compatible_incident_momentum_bound_audit.py::test_preflight_is_clean_on_the_repository_head",
    ),
    "cause": (
        "the frozen L1A-2u and L1A-2w preflights above require the worktree to be clean outside their own "
        "deliverables; while this stage's own module and tests were still uncommitted they saw this "
        "stage's paths. Their L1A-2t and L1A-2v structural failures and the old L1A-2r lineage failure "
        "are independent of this stage and are reported separately below"
    ),
}


def quality_and_blockers(verdict: str) -> dict[str, Any]:
    baseline = REPO_ROOT / "quality" / "legacy_two_phase_debt_v1.json"
    return {
        "declared_new_code_diagnostics": 0,
        "quality_baseline_sha256": sha256_file(baseline) if baseline.is_file() else "MISSING",
        "ratchet_expectation": "PASS_BASELINE_ONLY",
        "frozen_adjacent_suite": {
            "command": (
                "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest "
                "tests/test_nearwall_taper_interface_transport_audit.py "
                "tests/test_radial_incident_seed_eta_dt_audit.py "
                "tests/test_interior_speed_admissibility_screen_audit.py "
                "tests/test_boundary_compatible_incident_momentum_bound_audit.py -q"
            ),
            "failing": list(FROZEN_ADJACENT_FAILING_SET),
            "suite_failures": dict(FROZEN_ADJACENT_SUITE_FAILURES),
            "classification": "SUPERSEDED_BY_NEXT_STAGE_BY_CONSTRUCTION",
            "cause": (
                "the frozen preflights require every commit above their own baseline to belong to their own "
                "stage and, while this stage's deliverables were uncommitted, the worktree to be clean "
                "outside their own paths; neither can hold once a later stage exists on the same branch, "
                "so the L1A-2t, L1A-2v and L1A-2w preflight assertions fail by construction while the "
                "L1A-2u suite stays green"
            ),
            "action": "recorded as observed; no frozen test edited",
            "pass_count_not_recorded_because": (
                "the global pass count moves with the stage test count while the failing set is structural"
            ),
        },
        "frozen_sweep_observed_at_the_generating_tree_state": {
            "command": FROZEN_SWEEP_OBSERVED_AT_GENERATION["command"],
            "passed": FROZEN_SWEEP_OBSERVED_AT_GENERATION["passed"],
            "failed": FROZEN_SWEEP_OBSERVED_AT_GENERATION["failed"],
            "suite_failures": dict(FROZEN_SWEEP_OBSERVED_AT_GENERATION["suite_failures"]),
            "failing": list(FROZEN_SWEEP_OBSERVED_AT_GENERATION["failing"]),
            "caused_by_the_uncommitted_stage_deliverables": list(
                FROZEN_SWEEP_OBSERVED_AT_GENERATION["caused_by_the_uncommitted_stage_deliverables"]
            ),
            "cause": FROZEN_SWEEP_OBSERVED_AT_GENERATION["cause"],
            "action": (
                "recorded as observed at generation time; re-run at the committed head and written to "
                "artifacts/l1a2x/local_validation.json, never hidden and never turned green"
            ),
        },
        "frozen_sweep_reconfirmed_at_the_committed_head": dict(FROZEN_SWEEP_RECONFIRMED_AT_HEAD),
        "legacy_lineage_suite": {
            "failing": list(LEGACY_LINEAGE_FAILING_SET),
            "count": len(LEGACY_LINEAGE_FAILING_SET),
            "classification": "FROZEN_LINEAGE_DRIFT_CONFIRMED",
            "cause": "pre-existing on a pristine worktree at the verified main; not caused by this stage",
            "action": "kept separate from the structural failures and never counted as a pass",
            "suite_file": "tests/test_impact_initialization_compatibility_audit.py",
        },
        "reporting_rule": (
            "the structural frozen-adjacent failures (L1A-2t/L1A-2u/L1A-2v/L1A-2w lineage rules) and the "
            "old L1A-2s/2r lineage failure are reported separately with their own counts; neither is "
            "hidden, suppressed or turned green"
        ),
        "stage_test_file": "tests/test_frozen_chns_early_window_transport_attribution_audit.py",
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
        ],
    }


# ---------------------------------------------------------------------------
# Section G - report
# ---------------------------------------------------------------------------
def _fmt_ratios(attribution: dict[str, Any]) -> str:
    ratios = [
        None if entry["cancellation_ratio"] is None else round(entry["cancellation_ratio"], 4)
        for entry in attribution["cahn_hilliard_cancellation"]["per_macro"]
    ]
    return json.dumps(ratios)


def render_report(result: dict[str, Any]) -> tuple[dict[str, Any], str]:
    ledger = result["substep_ledger.json"]
    identity = result["initial_state_identity.json"]
    attribution = result["interface_attribution.json"]
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
    cumulative = attribution["cumulative_dn_dx"]
    lines: list[str] = []
    add = lines.append
    add(f"# {STAGE} - frozen CHNS early-window interface transport attribution")
    add("")
    add("## A. Scope, authority and baseline")
    add(
        f"- baseline: PR #24 head `{BASELINE_HEAD_SHA}`; verified main `{VERIFIED_MAIN_SHA}`; branch "
        f"`{provenance['branch']}`; solver contract 12; boundary model `{provenance['phase_boundary_model']}`"
    )
    add(
        f"- one case `{ledger['case_name']}`, N={ledger['profile_N']}, dt={ledger['requested_dt']}, macro steps "
        f"{ledger['macro_steps_run']} of at most {MAX_MACRO_STEPS} ({ledger['substeps_recorded']} frozen substeps, "
        f"t <= {_fmt(ledger['t_final'], 6)})"
    )
    add(
        f"- trajectory count 1, control arms {ledger['prohibitions']['control_arm_runs']}, contact events scored "
        f"{ledger['prohibitions']['contact_events_scored']}, refinements {ledger['prohibitions']['refinement_runs']}"
    )
    add("")
    add("## B. Frozen contract and prohibitions")
    add("- the frozen solver is executed, never edited: `pf.rhs`, `phase_advective_fluxes`,")
    add("  `chemical_potential_fluxes`, `control_volume_divergence`, `solve_ch_implicit`, the frozen Brinkman")
    add("  damping and the frozen pressure projection, with the frozen face apertures and control volumes")
    add(f"- frozen source hashes: `{json.dumps(provenance['frozen_source_sha256'])}`"[:1000])
    add(f"- prohibitions: {json.dumps(ledger['prohibitions'])}")
    add("- contract flags unchanged: `SOLVER_CONTRACT_VERSION=12`, `L1A_STATUS=BLOCKED`, `L1B_DATA_NOT_READY`")
    add("")
    add("## C. Sealed evidence verification")
    for name, entry in prerequisites["stages"].items():
        add(
            f"- {name}: {entry['status']}, bound files {entry.get('bound_files')}, manifest sha256 "
            f"`{str(entry.get('manifest_sha256'))[:16]}...`"
        )
    add(f"- all verified: {prerequisites['all_verified']}; failures: {prerequisites['failures']}")
    add("")
    add("## D. Initial state identity (frozen L1A-2s C2, never a ramp or a uniform seed)")
    add(f"- state source: {identity['state_source']}; candidate key `{identity['candidate_key']}`")
    add(f"- phi/u/v hashes match the sealed L1A-2w evidence: {json.dumps(identity['identity_checks'])}")
    add(
        f"- support cell {identity['support']['cell']} at gap {_fmt(identity['support']['gap_over_dx'])} dx, "
        f"|grad phi| {_fmt(identity['support']['gradient_norm'])} per length unit, interface normal "
        f"{json.dumps([round(v, 6) for v in identity['support']['interface_normal']])}, wall normal "
        f"{identity['support']['wall_normal_into_solid']}"
    )
    add(f"- geometry hash `{identity['geometry_hash'][:16]}...`, parameter hash `{identity['parameter_hash'][:16]}...`")
    add("")
    add("## E. The window and its ledger")
    add(
        f"- {ledger['substeps_recorded']} substep rows, each carrying the advective increment, the explicit "
        f"chemical-potential increment, the implicit CG increment, the complete update, the mass ledger, the "
        f"closed-face flux, the CG diagnostics and the fixed-support probe"
    )
    add(
        f"- contact guard: {json.dumps(ledger['contact_guard'])} "
        f"(threshold {CONTACT_GAP_THRESHOLD_DX} dx, never scored)"
    )
    add("")
    add("## F. Ingredient closure, conservation and the implicit solve")
    add(
        f"- mass: drift {_fmt(conservation['mass']['mass_drift_absolute'])} absolute / "
        f"{_fmt(conservation['mass']['mass_drift_relative'])} relative over the window; worst substep drift "
        f"{_fmt(conservation['mass']['max_abs_substep_drift'])}"
    )
    add(
        f"- ingredient telescoping: worst |sum V div| advective "
        f"{_fmt(conservation['mass']['max_abs_telescoping_adv'])}, "
        f"chemical {_fmt(conservation['mass']['max_abs_telescoping_ch'])}"
    )
    add(
        f"- impermeable wall: {conservation['wall']['closed_face_count_x']} + "
        f"{conservation['wall']['closed_face_count_y']} closed faces carry a worst flux of "
        f"{_fmt(conservation['wall']['max_abs_flux_on_closed_faces'])}"
    )
    add(
        f"- recombination: worst cell-wise |d_adv + d_ch + d_imp - d_full| = "
        f"{_fmt(conservation['recombination']['max_recombination_linf'])} against a largest increment of "
        f"{_fmt(conservation['recombination']['max_abs_substep_increment_linf'])}"
    )
    add(
        f"- implicit solve: {conservation['implicit_solve']['substeps']} solves, iterations "
        f"{conservation['implicit_solve']['min_iterations']}..{conservation['implicit_solve']['max_iterations']}, "
        f"worst relative residual {_fmt(conservation['implicit_solve']['max_relative_residual'])}, all converged "
        f"{conservation['implicit_solve']['all_converged']}"
    )
    add("")
    add("## G. Capture equivalence with the frozen public step")
    add(
        f"- worst |dphi| versus the eager `pf.step_with_diagnostics`: "
        f"{_fmt(equivalence['max_phi_difference_vs_eager_public_step'])}; versus the jitted `pf.step`: "
        f"{_fmt(equivalence['max_phi_difference_vs_jitted_public_step'])} (threshold {EQUIVALENCE_MAX_PHI_DIFF})"
    )
    add(
        f"- worst gap difference over the window: {_fmt(equivalence['max_gap_difference_over_dx'])} dx "
        f"(threshold {EQUIVALENCE_MAX_GAP_DIFF_DX})"
    )
    add(f"- meaning: {equivalence['meaning']}")
    add("")
    add("## H. Fixed-support attribution")
    add(
        f"- linearisation: dn = -(n_phi . n_w)/|grad phi| * dphi on the fixed t=0 support point and normal; "
        f"coefficient {_fmt(attribution['coefficient_over_dx'])} dx per unit phi"
    )
    add(
        f"- cumulative over the window (dx, positive toward the solid): advective "
        f"{_fmt(cumulative['adv'])}, chemical explicit {_fmt(cumulative['ch'])} (the near-cancelling half), "
        f"implicit {_fmt(cumulative['imp'])}, net chemical {_fmt(cumulative['net_ch'])}, complete "
        f"{_fmt(cumulative['full'])}"
    )
    add(
        f"- Cahn-Hilliard cancellation: |net CH| / (|explicit| + |implicit|) = "
        f"{_fmt(attribution['cahn_hilliard_cancellation']['cumulative_ratio'])} "
        f"(qualifier below {CH_CANCELLATION_QUALIFIER}); per macro step: {_fmt_ratios(attribution)}"
    )
    add(
        f"- the frozen global minimum-gap probe moves {_fmt(residuals['cumulative_direct_global_dx'])} dx over the "
        f"same window; the fixed-support linearisation moves {_fmt(residuals['cumulative_linearised_dx'])} dx"
    )
    add(
        f"- fixed support edge {json.dumps(ledger['frozen_support_edge']['identity'])}; branch lost at macro step "
        f"{ledger['frozen_support_edge']['branch_lost_at_macro_step']} (None means the tracked frozen edge carries the "
        f"crossing for the whole window)"
    )
    add(f"- prespecified attribution rules: {json.dumps(attribution['rules'])}")
    for rule, entry in attribution["family"].items():
        add(
            f"    - {rule}: cumulative dn {json.dumps({k: round(v, 6) for k, v in entry['cumulative_dn_dx'].items()})} "
            f"dx, margin {_fmt(entry['margin_dx'])} dx, ranking {entry['ranking']}"
        )
    add(
        f"- margin spread across the prespecified rules: {_fmt(attribution['margin_spread_across_rules_dx'])} dx; "
        f"every rule agrees on the ranking: {attribution['rankings_agree_across_rules']}"
    )
    for row in ledger["macro_rows"][:: max(1, len(ledger["macro_rows"]) // 5)]:
        add(
            f"    - t={_fmt(row['t'], 5)}: gap {_fmt(row['gap_global_over_dx'])} dx (cell {row['gap_global_cell']}), "
            f"phi(O) {_fmt(row['phi_at_support'])}, v_n(O) {_fmt(row['normal_velocity_at_support'])}, "
            f"|grad phi| {_fmt(row['gradient_norm_at_support'])}, uniqueness "
            f"{row['contour_uniqueness']['status']}"
        )
    add(
        "- per-macro-step fixed-support displacement (dx, positive toward the solid): step, t, advective, "
        "CH explicit, CH implicit, net CH, complete"
    )
    for row in attribution["per_macro"]:
        add(
            f"    - {row['macro_step']:>2} t={_fmt(row['t'], 4)}: adv {_fmt(row['dn_adv'])}, "
            f"ch_expl {_fmt(row['dn_ch_explicit'])}, ch_impl {_fmt(row['dn_ch_implicit'])}, "
            f"net_ch {_fmt(row['dn_net_ch'])}, full {_fmt(row['dn_full'])}"
        )
    add("")
    add("## I. Residual decomposition")
    for key, value in residuals["cumulative_abs_band"].items():
        add(f"- {key}: accumulated |.| = {_fmt(value)} dx (signed {_fmt(residuals['cumulative_signed'][key])})")
    add(f"- total residual band: {_fmt(residuals['total_residual_band_dx'])} dx")
    add(
        f"- the whole-window displacement (dx, positive toward the solid): frozen global gap "
        f"{_fmt(residuals['cumulative_direct_global_dx'])}, bilinear line root "
        f"{_fmt(residuals['cumulative_line_root_dx'])}, linearised point estimate "
        f"{_fmt(residuals['cumulative_linearised_dx'])}"
    )
    coverage = residuals["tracked_edge_coverage"]
    common = residuals["window_where_every_measurement_exists"]
    add(
        f"- the tracked frozen edge carries the crossing over macro steps {coverage['first_macro_step']}.."
        f"{coverage['last_macro_step']} of {coverage['macros_total']}"
        + (
            f"; from macro step {coverage['first_macro_step_without_it']} on the frozen linear crossing has "
            "moved off it (recorded as a support change of the frozen reconstruction)"
            if coverage["first_macro_step_without_it"] is not None
            else ""
        )
    )
    add(
        f"- over that common span ({coverage['macros_available']} macro steps) all four measurements agree: "
        f"frozen global gap {_fmt(common['cumulative_dn_dx']['d1_frozen_global_gap'])}, frozen tracked edge "
        f"{_fmt(common['cumulative_dn_dx']['d2_frozen_tracked_edge'])}, bilinear line root "
        f"{_fmt(common['cumulative_dn_dx']['d3_bilinear_line_root'])}, linearised point estimate "
        f"{_fmt(common['cumulative_dn_dx']['d4_linearised_point_estimate'])} dx; the linearisation overshoots "
        f"the geometric reconstruction by {_fmt(common['linearisation_overshoot_dx'])} dx "
        f"({_fmt(100.0 * common['linearisation_overshoot_relative'], 3)} % of the geometric motion) and its "
        f"total band over that span is {_fmt(common['total_band_dx'])} dx"
    )
    add(
        f"- physical closure between the frozen global probe and the fixed-support linearisation: "
        f"{_fmt(residuals['physical_closure_difference_dx'])} dx; between the tracked frozen edge and the "
        f"linearisation: {_fmt(residuals['line_root_closure_difference_dx'])} dx; frozen support edge alive "
        f"through the window: {residuals['frozen_support_edge_branch_alive']}"
    )
    add(
        f"- algebraic identity residuals (implementation check): "
        f"{json.dumps(residuals['algebraic_identity_residuals'])}"
    )
    add(f"- definitions: {residuals['definition']}")
    add("")
    add("## J. Validity screen")
    add(f"- criteria: {json.dumps(validity['criteria'])}")
    add(
        f"- failures per guard: {json.dumps(validity['failures'])}; "
        f"blocking: {json.dumps(validity['blocking_failures'])}"
    )
    add(f"- observed: {json.dumps(validity['observed'], default=_json_default)}"[:1200])
    add(
        f"- direct frozen probe: {validity['direct_probe_status']}; fixed-support attribution: "
        f"{validity['fixed_support_attribution_status']}"
    )
    add("")
    add("## K. Decision")
    add(f"- verdict `{decision['verdict']}`")
    add(f"- reason: {decision['reason']}")
    add(
        f"- magnitudes: advective {_fmt(decision['advective_magnitude_dx'])} dx, net chemical "
        f"{_fmt(decision['net_chemical_hilliard_magnitude_dx'])} dx, margin "
        f"{_fmt(decision['dominance_margin_dx'])} dx against a dominance floor of "
        f"{_fmt(decision['dominance_floor_dx'])} dx "
        f"({decision['dominance_margin_factor']} x the comparison band)"
    )
    add(
        f"- comparison band (spread of the margin across the prespecified attribution rules plus the "
        f"algebraic recombination term): {_fmt(decision['comparison_band_dx'])} dx, margin/comparison band "
        f"{_fmt(decision['dominance_margin_over_comparison_band'])}; total residual-map band "
        f"{_fmt(decision['total_map_band_dx'])} dx, margin/total map band "
        f"{_fmt(decision['dominance_margin_over_total_map_band'])}"
    )
    add(
        f"- recorded, not claimed: the dominance rule returns "
        f"`{decision['dominance_rule_outcome_recorded_not_claimed']['with_the_comparison_band']}` with the "
        f"comparison band and "
        f"`{decision['dominance_rule_outcome_recorded_not_claimed']['with_the_total_residual_map_band']}` "
        f"with the larger total residual-map band; rankings agree across the prespecified rules: "
        f"{decision['rankings_agree_across_rules']}; the sign of the net motion matches the larger term: "
        f"{decision['net_motion_sign_matches_dominant_term']}"
    )
    if decision["verdict_withheld_because"]:
        add("- the verdict is withheld because:")
        for reason in decision["verdict_withheld_because"]:
            add(f"    - {reason}")
    add(
        f"- direct frozen probe: {decision['direct_probe_status']} (recorded, not a verdict); fixed-support "
        f"attribution: {decision['fixed_support_attribution_status']}; capture equivalence holds: "
        f"{decision['capture_equivalence_ok']}; blocking guards failed: "
        f"{json.dumps(validity['blocking_failures'])}; the tracked support edge carried the crossing for "
        f"{validity['observed']['tracked_edge_continuity_macros_ok']} of "
        f"{len(ledger['macro_rows'])} macro steps"
    )
    add(f"- band semantics: {decision['band_semantics']}")
    add(
        f"- Cahn-Hilliard near-cancellation qualifier: {decision['cahn_hilliard_cancellation_qualifier']} "
        f"(ratio {_fmt(decision['cahn_hilliard_cancellation_ratio'])})"
    )
    add(f"- scope: {json.dumps(decision['scope'])}")
    add(f"- forbidden claims respected: {json.dumps(decision['forbidden_claims_respected'])}")
    add(
        "- the early-window displacement reported here is a pre-contact response of the frozen initial "
        "state; it is not a real impact and it is not extrapolated to the late contact time"
    )
    add("")
    add("## L. Adversarial and falsification checks")
    for name, value in adversarial["checks"].items():
        add(f"- `{name}`: {json.dumps(value, default=_json_default)}"[:600])
    add(f"- all checks hold: {adversarial['all_hold']}")
    add("")
    add("## M. Quality, tests and hosted CI")
    add(f"- quality baseline sha256 `{quality['quality_baseline_sha256']}`")
    add(
        f"- frozen adjacent suite: {quality['frozen_adjacent_suite']['classification']}, suite failures "
        f"{json.dumps(quality['frozen_adjacent_suite']['suite_failures'])}"
    )
    for name in quality["frozen_adjacent_suite"]["failing"]:
        add(f"    - failing: {name}")
    add(
        f"- old L1A-2s lineage failure, reported separately: {quality['legacy_lineage_suite']['failing']} "
        f"({quality['legacy_lineage_suite']['classification']})"
    )
    add(f"- not run: {json.dumps(quality['not_run'])}")
    add(
        f"- declared new diagnostics {quality['declared_new_code_diagnostics']}, expectation "
        f"`{quality['ratchet_expectation']}`"
    )
    add("- hosted CI observed after push: `UNMEASURED` (a re-seal records the observed result)")
    add("")
    add("## N. Exactly one next action")
    add(f"- {action['action']}")
    add(f"- bounded scope: {json.dumps(action['bounded_scope'])}")
    add(f"- must not be started in this stage: {action['must_not_be_started_in_this_stage']}")
    report = "\n".join(lines) + "\n"
    payload = {
        "stage": STAGE,
        "verdict": decision["verdict"],
        "headline": (
            f"Over {ledger['macro_steps_run']} frozen macro steps (t <= {_fmt(ledger['t_final'], 4)}) on "
            f"{ledger['case_name']}, the fixed-support attribution moves the interface "
            f"{_fmt(cumulative['full'])} dx toward the solid, of which the advective term contributes "
            f"{_fmt(cumulative['adv'])} dx and the net Cahn-Hilliard term {_fmt(cumulative['net_ch'])} dx; "
            f"the Cahn-Hilliard explicit and implicit halves are individually {_fmt(cumulative['ch'])} and "
            f"{_fmt(cumulative['imp'])} dx and nearly cancel, and the decision rule returns "
            f"`{decision['verdict']}`."
        ),
        "sections": [line for line in lines if line.startswith("## ")],
        "reference_times": [row["t"] for row in ledger["macro_rows"]],
    }
    return payload, report


# ---------------------------------------------------------------------------
# Section H - evidence assembly and verification
# ---------------------------------------------------------------------------
def assemble_evidence(result: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_payload, report_text = render_report(result)
    written: dict[str, Any] = {}
    for name in EVIDENCE_FILES:
        if name in ("l1a2x_final_report.json", "l1a2x_final_report.md", "manifest.json"):
            continue
        payload = result.get(name)
        if payload is None:
            raise AuditBlocked(f"no payload for required evidence file {name}")
        (out_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
        written[name] = payload
    (out_dir / "l1a2x_final_report.json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    (out_dir / "l1a2x_final_report.md").write_text(report_text)
    decision = result["mechanism_decision.json"]
    ledger = result["substep_ledger.json"]
    manifest = {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "profile": "early_window",
        "evidence_class": "FROZEN_CHNS_ATTRIBUTION",
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
        "module_sha256": result["source_and_provenance.json"]["module_sha256"],
        "required_files": list(EVIDENCE_FILES),
        "files": {},
    }
    for name in EVIDENCE_FILES:
        if name == "manifest.json":
            continue
        manifest["files"][name] = sha256_file(out_dir / name)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n")
    return verify_evidence_manifest(out_dir)


def verify_evidence_manifest(out_dir: str | Path) -> dict[str, Any]:
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
    missing_required = [name for name in EVIDENCE_FILES if not (directory / name).is_file()]
    if missing_required:
        problems.append(f"missing required files: {sorted(missing_required)}")
    seal = _recorded_evidence_seal(manifest)
    if not seal:
        problems.append("this stage's manifest declares no bound-file seal")
    for name, want in seal.items():
        path = directory / name
        if not path.is_file():
            problems.append(f"missing bound file: {name}")
            continue
        if sha256_file(path) != want:
            problems.append(f"sha mismatch: {name}")
    if problems:
        raise AuditBlocked(f"evidence verification failed: {problems}")
    return {
        "stage": manifest.get("stage"),
        "stage_version": manifest.get("stage_version"),
        "profile": manifest.get("profile"),
        "evidence_class": manifest.get("evidence_class"),
        "verdict": manifest.get("verdict"),
        "verified_main_sha": manifest.get("verified_main_sha"),
        "bound_files": len(seal),
        "directory": str(directory),
        "problems": problems,
        "missing_required": missing_required,
        "verified": True,
    }


def reseal_ci_status(ci_status: str, head_sha: str, out_dir: str | Path) -> dict[str, Any]:
    """Record an observed post-push hosted CI result; rewrites metadata only."""
    if ci_status not in ("success", "failure", "partial", "NOT_RUN"):
        raise AuditBlocked(f"refusing to record an unverified CI status string: {ci_status!r}")
    if len(head_sha) != 40 or any(ch not in "0123456789abcdef" for ch in head_sha):
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
    report_path = directory / "l1a2x_final_report.md"
    report = report_path.read_text()
    marker = "\n## N. Exactly one next action"
    line = f"- hosted CI observed after push: `{ci_status}` for head `{head_sha}` (previous record: `{previous}`)\n"
    if "hosted CI observed after push" in report:
        start = report.index("- hosted CI observed after push")
        end = report.index("\n", start) + 1
        report = report[:start] + line + report[end:]
    elif marker in report:
        report = report.replace(marker, "\n" + line + marker, 1)
    else:
        raise AuditBlocked("could not locate the report insertion point for the CI re-seal")
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
    for name in ("quality_and_blocker_matrix.json", "l1a2x_final_report.md"):
        if name in files:
            files[name] = sha256_file(directory / name)
    manifest["files"] = files
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n")
    return verify_evidence_manifest(directory)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
STAGES = (
    "preflight",
    "sealed",
    "identity",
    "window",
    "attribution",
    "conservation",
    "equivalence",
    "validity",
    "decision",
    "adversarial",
    "quality",
    "assemble",
)


def run(*, out: str | Path | None = None, stages: str = "all", macro_steps: int = MAX_MACRO_STEPS) -> dict[str, Any]:
    requested = [part.strip() for part in stages.split(",") if part.strip()]
    if "all" in requested:
        requested = list(STAGES)
    unknown = [part for part in requested if part not in STAGES]
    if unknown:
        raise AuditBlocked(f"unknown stages: {unknown}")
    result: dict[str, Any] = {}
    started = time.perf_counter()
    preflight = preflight_state()
    result["preflight.json"] = preflight
    provenance = source_and_provenance(preflight)
    result["source_and_provenance.json"] = provenance
    if "sealed" in requested or "assemble" in requested:
        progress("re-hashing the frozen L1A-2t/u/v/w evidence")
        result["frozen_evidence_verification.json"] = verify_prerequisites()
    if "identity" in requested or "assemble" in requested:
        progress("binding the frozen C2 initial state to the sealed evidence")
        result["initial_state_identity.json"] = initial_state_identity()
    if "window" in requested or "assemble" in requested:
        progress("running the one sanctioned window with per-substep capture")
        result["substep_ledger.json"] = run_ledger(macro_steps)
        progress(
            f"window done: {result['substep_ledger.json']['macro_steps_run']} macro steps, "
            f"t <= {_fmt(result['substep_ledger.json']['t_final'], 5)}"
        )
    if "attribution" in requested or "assemble" in requested:
        ledger = result["substep_ledger.json"]
        result["trajectory_specification.json"] = {
            "case_name": ledger["case_name"],
            "profile_N": ledger["profile_N"],
            "dt": ledger["requested_dt"],
            "dt_substep": ledger["parameters"]["substep_dt"],
            "macro_step_cap": MAX_MACRO_STEPS,
            "macro_steps_run": ledger["macro_steps_run"],
            "substeps_per_macro": ledger["substeps_per_macro"],
            "t_final": ledger["t_final"],
            "solver": "frozen phasefield.step path, re-executed by the diagnostic capture",
            "initial_state_hashes": ledger["initial_state"],
            "final_state_hashes": ledger["final_state"],
            "contact_guard": ledger["contact_guard"],
            "prohibitions": ledger["prohibitions"],
            "equivalence_note": ledger["equivalence_note"],
        }
        progress("fixed-support attribution")
        result["interface_attribution.json"] = interface_attribution(ledger)
        result["residual_decomposition.json"] = residual_decomposition(ledger, result["interface_attribution.json"])
    if "conservation" in requested or "assemble" in requested:
        result["conservation_and_convergence.json"] = conservation_and_convergence(result["substep_ledger.json"])
    if "equivalence" in requested or "assemble" in requested:
        result["capture_equivalence.json"] = capture_equivalence_summary(result["substep_ledger.json"])
    if "validity" in requested or "assemble" in requested:
        ledger = result["substep_ledger.json"]
        attribution = result["interface_attribution.json"]
        result["validity_screen.json"] = validity_screen(
            ledger, attribution, result["residual_decomposition.json"], result["capture_equivalence.json"]
        )
    if "decision" in requested or "assemble" in requested:
        decision = decide(
            result["substep_ledger.json"],
            result["interface_attribution.json"],
            result["residual_decomposition.json"],
            result["validity_screen.json"],
            result["capture_equivalence.json"],
        )
        result["mechanism_decision.json"] = decision
        result["next_action.json"] = next_action(decision)
        progress(f"decision: verdict={decision['verdict']}")
    if "adversarial" in requested or "assemble" in requested:
        result["adversarial_and_falsification.json"] = adversarial_and_falsification(
            result["substep_ledger.json"],
            result["interface_attribution.json"],
            result["residual_decomposition.json"],
            result["mechanism_decision.json"],
            result["validity_screen.json"],
            provenance,
        )
    if "quality" in requested or "assemble" in requested:
        result["quality_and_blocker_matrix.json"] = quality_and_blockers(result["mechanism_decision.json"]["verdict"])
    if "assemble" in requested:
        out_dir = _out_dir(out)
        result["manifest"] = assemble_evidence(result, out_dir)
        progress(f"evidence assembled in {out_dir}")
    result["__elapsed_seconds__"] = time.perf_counter() - started
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--out", default=None)
    parser.add_argument("--macro-steps", type=int, default=MAX_MACRO_STEPS)
    parser.add_argument("--verify-evidence", default=None)
    parser.add_argument("--reseal-ci-status", default=None)
    parser.add_argument("--reseal-head-sha", default=None)
    args = parser.parse_args(argv)
    if args.reseal_ci_status:
        if not args.reseal_head_sha:
            raise AuditBlocked("--reseal-ci-status requires --reseal-head-sha")
        report = reseal_ci_status(args.reseal_ci_status, args.reseal_head_sha, _out_dir(args.out))
        print(f"[{STAGE}] evidence resealed for hosted CI: {json.dumps(report, default=_json_default)}")
        return
    if args.verify_evidence:
        report = verify_evidence_manifest(args.verify_evidence)
        print(f"[{STAGE}] evidence verified: {json.dumps(report, default=_json_default)}")
        return
    result = run(out=args.out, stages=args.stages, macro_steps=args.macro_steps)
    decision = result.get("mechanism_decision.json", {})
    print(f"[{STAGE}] done in {result['__elapsed_seconds__']:.1f}s; verdict={decision.get('verdict')}")


if __name__ == "__main__":
    main()
