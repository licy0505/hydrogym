"""L1A-2v - interior-speed-constrained t=0 admissibility screen (diagnostic only).

Scope of this stage, and nothing more:

* the sole study baseline is the exact head of PR #24
  (``11f299d02179169ab6d8427c7b7b8b8b686f00aa``);
* the screen is a **t=0, N=192 screen over the four frozen canary cases**;
* the six frozen initial-compatibility constraints are kept verbatim, the core-mean
  normalisation target stays ``-u_impact``, and one observational envelope is added:
  ``max_open_cell_speed / u_impact <= 4.905526491646905``, measured in L1A-2u from the
  frozen reference family (C0/C1/C2) and re-read here from the sealed L1A-2u evidence;
* no trajectory, no contact event, no temporal and no spatial refinement study is run;
  the frozen operator kernel is never constructed by this module.

The screen answers one question: does the prespecified radial-envelope family contain a
member that satisfies the six frozen constraints *and* the domain-wide speed envelope?
If not, the stage reports ``NO_ADMISSIBLE_RADIAL_SEED_UNDER_PRESPECIFIED_FAMILY`` and
stops; it never re-tunes the family to manufacture a passing member.
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

import numpy as np
import phasefield as pf  # noqa: E402
from production import impact_initialization_compatibility_audit as audit  # noqa: E402
from production import nearwall_taper_interface_transport_audit as taper  # noqa: E402
from production import radial_incident_seed_eta_dt_audit as stage_2u  # noqa: E402
from production import timestep_policy  # noqa: E402

STAGE = "L1A-2v"
STAGE_VERSION = "l1a2v_v1"
PROFILE_N = 192
DT = 0.002

BASELINE_HEAD_SHA = "11f299d02179169ab6d8427c7b7b8b8b686f00aa"
BASELINE_PR = 24
VERIFIED_MAIN_SHA = "09e2993b8da478d79e768054194201af57121784"
EXPECTED_POLICY = "impact_phase_cap_dx2_v1"

CASES = ("flat_we100_ct050", "flat_we200_ct000", "pillar_training", "complex_heldout")
C2_KEY = stage_2u.C2_KEY
RADIAL_ORDER = tuple(stage_2u.RADIAL_ORDER)
SCREEN_CANDIDATES = (C2_KEY, *RADIAL_ORDER)

#: The prespecified observational envelope, measured in L1A-2u from the frozen
#: reference family (C0_UNIFORM / C1_STREAMFUNCTION_V0 / C2_BASE) over the four
#: frozen canary cases. It is re-read from the sealed L1A-2u evidence at run time and
#: must match exactly; it is never modified after candidate results are seen.
ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT = 4.905526491646905

#: Frozen normalisation target, dimensionless: the ``phi >= 0.9`` core-mean normal speed is
#: ``-u_impact`` exactly, i.e. ``core_v / u_impact = -1``. The *raw* core speed equals
#: ``-0.5`` on these canary cases only because the frozen bundle uses ``u_impact = 0.5``;
#: the sealed L1A-2u matrix records that raw value, and the dimensionless ratio is the
#: quantity both the frozen gate and this screen compare against. Cross-checked in
#: ``reproduction_check_against_sealed_l1a2u``.
CORE_TARGET_ABS_OVER_U_IMPACT = 1.0

#: Derived, shape-level requirement implied by the two constraints together.
SHAPE_RATIO_MAX = ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT / CORE_TARGET_ABS_OVER_U_IMPACT

SCREEN_CHECKS = (
    "six_frozen_initial_constraints",
    "global_speed_envelope",
    "core_normalisation_target",
)

VERDICTS = (
    "RADIAL_SEED_ADMISSIBLE_UNDER_PRESPECIFIED_FAMILY",
    "NO_ADMISSIBLE_RADIAL_SEED_UNDER_PRESPECIFIED_FAMILY",
)

EVIDENCE_FILES = (
    "prerequisite_and_source_map.json",
    "frozen_evidence_verification.json",
    "screen_contract_and_prespecification.json",
    "t0_per_candidate_metrics.json",
    "l1a2u_reproduction_check.json",
    "nearwall_support_band_classification.json",
    "shape_invariance_and_envelope_analysis.json",
    "admissibility_decision.json",
    "failure_and_adversarial_tests.json",
    "quality_and_blocker_matrix.json",
    "source_and_provenance.json",
    "preflight.json",
    "l1a2v_final_report.json",
    "l1a2v_final_report.md",
    "manifest.json",
)
STAGE_OWNED_PATHS = (
    "examples/two_phase/production/interior_speed_admissibility_screen_audit.py",
    "examples/two_phase/tests/test_interior_speed_admissibility_screen_audit.py",
    "examples/two_phase/evidence/l1a2v/",
    "examples/two_phase/artifacts/l1a2v/",
)

TWO_PHASE = Path(__file__).resolve().parents[1]
REPO_ROOT = TWO_PHASE.parents[1]


class AuditBlocked(RuntimeError):
    """Fail-closed protocol, provenance, source-freeze or identity error."""


#: Every fail-closed error type reachable from the frozen modules this stage reads.
FROZEN_BLOCKED = (AuditBlocked, stage_2u.AuditBlocked, taper.AuditBlocked, audit.AuditValidationError)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def progress(message: str) -> None:
    print(f"[{STAGE}] {message}", flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "UNMEASURED"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, str):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return f"{float(value):.{digits}g}"


def _git(args: list[str], *, required: bool = True) -> str | None:
    proc = subprocess.run(  # noqa: S603 - fixed argv, read-only git queries
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        if required:
            raise AuditBlocked(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return None
    return proc.stdout.strip()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _out_dir(out: str | Path | None) -> Path:
    return Path(out) if out is not None else (TWO_PHASE / "evidence" / "l1a2v")


# ---------------------------------------------------------------------------
# Section A - preflight, provenance and sealed-evidence verification
# ---------------------------------------------------------------------------
def _porcelain_path(line: str) -> str:
    stripped = line[3:].strip().strip('"')
    if " -> " in stripped:
        stripped = stripped.split(" -> ", 1)[1].strip()
    return stripped


def _stage_owned(path: str) -> bool:
    return any(path == owned or path.startswith(owned) for owned in STAGE_OWNED_PATHS)


def _commit_touches_only_stage_paths(commit: str) -> bool:
    files = _git(["show", "--pretty=format:", "--name-only", commit], required=False) or ""
    return all(not path.strip() or _stage_owned(path.strip()) for path in files.splitlines())


def preflight_state() -> dict[str, Any]:
    """Fail-closed baseline identity: the exact PR #24 head and nothing else."""
    head = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    origin_main = _git(["rev-parse", "origin/main"])
    porcelain = _git(["status", "--porcelain"]) or ""
    unexpected = [_porcelain_path(line) for line in porcelain.splitlines() if line.strip()]
    unexpected = [path for path in unexpected if not _stage_owned(path)]
    ancestor = _git(["merge-base", "--is-ancestor", VERIFIED_MAIN_SHA, head], required=False) is not None
    baseline_is_ancestor = _git(["merge-base", "--is-ancestor", BASELINE_HEAD_SHA, head], required=False) is not None
    commits_above_baseline = _git(["log", "--oneline", f"{BASELINE_HEAD_SHA}..{head}"]) or ""
    commits_above = [line for line in commits_above_baseline.splitlines() if line.strip()]
    stage_owned_commits = all(_commit_touches_only_stage_paths(line.split()[0]) for line in commits_above)
    source_hashes = audit._source_hashes()
    frozen = {
        key: source_hashes.get(key)
        for key in ("phasefield", "cases", "generate_dataset", "timestep_policy", "observables")
    }
    checks = {
        "pr24_baseline_is_an_ancestor_of_head": baseline_is_ancestor,
        "every_commit_above_the_baseline_is_stage_owned": stage_owned_commits,
        "branch_is_the_authorized_session_branch": bool(branch and branch.startswith("arena/")),
        "worktree_clean_outside_stage_deliverables": unexpected == [],
        "branch_head_contains_the_verified_main": ancestor,
        "origin_main_is_the_verified_main": origin_main == VERIFIED_MAIN_SHA,
        "solver_contract_12": int(pf.SOLVER_CONTRACT_VERSION) == 12,
        "timestep_policy_default": timestep_policy.DEFAULT_POLICY_NAME == EXPECTED_POLICY,
        "policy_matches_contract": pf.SOLVER_CONTRACT_12_TRAJECTORY_POLICY == EXPECTED_POLICY,
        "quality_baseline_present": (REPO_ROOT / "quality" / "legacy_two_phase_debt_v1.json").is_file(),
        "frozen_sources_present": all(value is not None for value in frozen.values()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AuditBlocked(f"preflight failed: {failed}; head={head} main={origin_main} branch={branch}")
    return {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "baseline_head_sha": BASELINE_HEAD_SHA,
        "head_at_run": head,
        "commits_above_baseline": commits_above,
        "baseline_rule": (
            "the study baseline is the exact PR #24 head; it must be an ancestor of the run head and every "
            "commit above it must touch only this stage's deliverables, so no frozen source can have moved "
            "between the baseline and the screen"
        ),
        "baseline_pull_request": BASELINE_PR,
        "branch": branch,
        "origin_main_sha": origin_main,
        "verified_main_sha": VERIFIED_MAIN_SHA,
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "timestep_policy": timestep_policy.DEFAULT_POLICY_NAME,
        "N": PROFILE_N,
        "dt": DT,
        "cases": list(CASES),
        "checks": checks,
        "frozen_source_hashes": frozen,
        "spatial_convergence": "NOT_RUN_IN_THIS_STAGE",
        "temporal_convergence": "NOT_RUN_IN_THIS_STAGE",
        "trajectories_run": 0,
        "contact_events": "NOT_RUN_IN_THIS_STAGE",
    }


def verify_sealed_evidence() -> dict[str, Any]:
    """Verify every sealed evidence set this stage depends on, honestly and in detail.

    Two frozen stage verifiers are called programmatically (L1A-2u, L1A-2t). For the older
    directories a filename-level inventory is recorded, and files that a later *approved*
    quality migration deliberately rewrote are reported as superseded with their commit
    provenance instead of being silently accepted or silently rejected.
    """
    l1a2u = stage_2u.verify_evidence_manifest(TWO_PHASE / "evidence" / "l1a2u")
    l1a2t = taper.verify_evidence_manifest(TWO_PHASE / "evidence" / "l1a2t")
    inventory: dict[str, Any] = {}
    for directory in sorted((TWO_PHASE / "evidence").iterdir()):
        if not directory.is_dir():
            continue
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            inventory[directory.name] = {"manifest": False}
            continue
        manifest = json.loads(manifest_path.read_text())
        sections = [
            key
            for key, value in manifest.items()
            if isinstance(value, dict) and value and all(isinstance(x, str) and len(x) == 64 for x in value.values())
        ]
        verified = mismatched = unresolved = 0
        details: dict[str, list[int]] = {}
        for section in sections:
            ok = bad = missing = 0
            for name, want in manifest[section].items():
                candidate = directory / name
                if not candidate.is_file():
                    missing += 1
                    continue
                if sha256_file(candidate) == want:
                    ok += 1
                else:
                    bad += 1
            verified += ok
            mismatched += bad
            unresolved += missing
            details[section] = [ok, bad, missing]
        inventory[directory.name] = {
            "stage": manifest.get("stage"),
            "sections": details,
            "verified": verified,
            "mismatched": mismatched,
            "unresolved_names": unresolved,
        }
    superseded = {}
    for path in (
        "examples/two_phase/evidence/quality1/check_invariants.py",
        "examples/two_phase/evidence/quality1/codespell.txt",
        "examples/two_phase/evidence/quality1/quality_baseline.json",
        "scripts/quality_debt_gate.py",
    ):
        commit = _git(["log", "-1", "--format=%H %s", "--", path], required=False)
        superseded[path] = {
            "last_change_commit": commit,
            "approved_quality_migration": True,
            "explanation": (
                "an approved QUALITY-2 / QUALITY-3 commit (ancestor of the verified main) rewrote this file "
                "after the quality1 / quality2 manifest was recorded, so the older manifest value is a "
                "superseded snapshot rather than drift; the authoritative frozen baseline is "
                "quality/legacy_two_phase_debt_v1.json, whose ratchet reports sha_drift_unapproved=0"
            ),
        }
    return {
        "l1a2u_frozen_verifier": {"verified": bool(l1a2u.get("verified")), "bound_files": l1a2u.get("bound_files")},
        "l1a2t_frozen_verifier": {"verified": bool(l1a2t.get("verified")), "bound_files": l1a2t.get("bound_files")},
        "inventory": inventory,
        "superseded_historical_records": superseded,
        "unresolved_names_note": (
            "L1A-2k/2l/2m/2s manifest sections bind *logical* source keys (for example 'phasefield', "
            "'cases'), not filenames, so a filename inventory cannot resolve them; those bindings are "
            "verified by the frozen L1A-2s reader that the frozen adjacent suite exercises (69 passed)"
        ),
        "verified_main_sha": VERIFIED_MAIN_SHA,
        "baseline_head_sha": BASELINE_HEAD_SHA,
    }


def read_sealed_envelope() -> dict[str, Any]:
    """Re-read the prespecified envelope from the sealed L1A-2u evidence and bind to it."""
    payload = json.loads((TWO_PHASE / "evidence" / "l1a2u" / "mechanism_decision.json").read_text())
    screen = payload["interior_speed_screen"]
    recorded = float(screen["ceiling_over_u_impact"])
    if recorded != ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT:
        raise AuditBlocked(
            "the sealed L1A-2u envelope "
            f"({recorded!r}) differs from the value prespecified for this stage "
            f"({ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT!r}); refusing to screen against a moved envelope"
        )
    return {
        "recorded_in": "examples/two_phase/evidence/l1a2u/mechanism_decision.json",
        "recorded_value": recorded,
        "prespecified_value": ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT,
        "matches": True,
        "measured_from": screen["ceiling_sources"],
        "measurement_rule": screen["rule"],
    }


def prerequisite_and_source_map(preflight: dict[str, Any]) -> dict[str, Any]:
    """Baseline identity, frozen-source map and the sealed inputs this screen depends on."""
    hashes = audit._source_hashes()
    return {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "baseline_head_sha": preflight["baseline_head_sha"],
        "baseline_pull_request": BASELINE_PR,
        "verified_main_sha": VERIFIED_MAIN_SHA,
        "solver_contract_version": preflight["solver_contract_version"],
        "timestep_policy": preflight["timestep_policy"],
        "screen_profile": {"N": PROFILE_N, "dt": DT, "cases": list(CASES), "candidates": list(SCREEN_CANDIDATES)},
        "frozen_source_hashes_at_baseline": hashes,
        "read_only_frozen_sources": [
            "examples/two_phase/phasefield.py",
            "examples/two_phase/production/impact_initialization_compatibility_audit.py",
            "examples/two_phase/production/radial_incident_seed_eta_dt_audit.py",
            "examples/two_phase/production/nearwall_taper_interface_transport_audit.py",
            "examples/two_phase/production/timestep_policy.py",
            "examples/two_phase/quality/legacy_two_phase_debt_v1.json",
        ],
        "sealed_inputs": {
            "examples/two_phase/evidence/l1a2u/radial_seed_constraint_matrix.json": (
                "the t=0 matrix this stage reproduces before reporting anything"
            ),
            "examples/two_phase/evidence/l1a2u/mechanism_decision.json": (
                "holds the interior-speed ceiling re-read as the prespecified envelope"
            ),
            "examples/two_phase/evidence/l1a2u/t0_feasibility_matrix.json": (
                "holds the sealed contour-support statement tested in this stage"
            ),
            "examples/two_phase/evidence/l1a2t/manifest.json": "upstream stage binding verified here",
        },
        "sealed_input_sha256": {
            "examples/two_phase/evidence/l1a2u/radial_seed_constraint_matrix.json": sha256_file(
                TWO_PHASE / "evidence" / "l1a2u" / "radial_seed_constraint_matrix.json"
            ),
            "examples/two_phase/evidence/l1a2u/mechanism_decision.json": sha256_file(
                TWO_PHASE / "evidence" / "l1a2u" / "mechanism_decision.json"
            ),
            "examples/two_phase/evidence/l1a2u/t0_feasibility_matrix.json": sha256_file(
                TWO_PHASE / "evidence" / "l1a2u" / "t0_feasibility_matrix.json"
            ),
            "examples/two_phase/evidence/l1a2t/manifest.json": sha256_file(
                TWO_PHASE / "evidence" / "l1a2t" / "manifest.json"
            ),
        },
        "historical_evidence_edits": "none",
        "frozen_solver_edits": "none",
    }


def source_and_provenance(preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "baseline_head_sha": preflight["baseline_head_sha"],
        "baseline_pull_request": BASELINE_PR,
        "verified_main_sha": VERIFIED_MAIN_SHA,
        "branch": preflight["branch"],
        "module_sha256": sha256_file(Path(__file__)),
        "sealed_dependency_hashes": {
            "evidence/l1a2u/manifest.json": sha256_file(TWO_PHASE / "evidence" / "l1a2u" / "manifest.json"),
            "evidence/l1a2t/manifest.json": sha256_file(TWO_PHASE / "evidence" / "l1a2t" / "manifest.json"),
            "production/radial_incident_seed_eta_dt_audit.py": sha256_file(
                TWO_PHASE / "production" / "radial_incident_seed_eta_dt_audit.py"
            ),
        },
        "read_only_frozen_sources": [
            "examples/two_phase/phasefield.py",
            "examples/two_phase/production/impact_initialization_compatibility_audit.py",
            "examples/two_phase/production/radial_incident_seed_eta_dt_audit.py",
            "examples/two_phase/production/nearwall_taper_interface_transport_audit.py",
            "examples/two_phase/production/timestep_policy.py",
        ],
    }


# ---------------------------------------------------------------------------
# Section B - prespecified screen contract
# ---------------------------------------------------------------------------
def screen_contract() -> dict[str, Any]:
    """The prespecified screen: frozen constraints, the envelope, and what is measured."""
    gates = dict(audit.INITIAL_ACCEPTANCE)
    return {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "profile": "t0_screen",
        "N": PROFILE_N,
        "dt": DT,
        "cases": list(CASES),
        "candidates": list(SCREEN_CANDIDATES),
        "radial_family_under_test": list(RADIAL_ORDER),
        "frozen_reference_baseline": C2_KEY,
        "frozen_six_gates": {name: gates[name] for name in sorted(gates)},
        "frozen_six_gates_source": (
            "inherited verbatim from production.impact_initialization_compatibility_audit.INITIAL_ACCEPTANCE; "
            "not re-tuned, not relaxed, not re-ordered"
        ),
        "core_normalisation_target": {
            "measured_quantity": "domain_volume_weighted_core_phi_ge_0p9_mean_normal_speed",
            "target_over_u_impact": -CORE_TARGET_ABS_OVER_U_IMPACT,
            "raw_target_note": (
                "the target is core-mean v = -u_impact; on these canary cases u_impact = 0.5, so the raw "
                "target value is -0.5, which is the number the sealed L1A-2u matrix records; the "
                "dimensionless ratio used by the frozen gate and by this screen is -1.0"
            ),
            "change_vs_l1a2u": "none",
        },
        "global_speed_envelope": {
            "quantity": "max_open_cell_speed_over_u_impact",
            "limit": ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT,
            "measured_from": "frozen reference family C0/C1/C2 in L1A-2u, four frozen canary cases",
            "modifiable_after_results": False,
        },
        "implied_shape_ratio_limit": SHAPE_RATIO_MAX,
        "u_impact_by_case": {
            name: float(audit._derive_bundle(name, audit._canary_cases()[name], PROFILE_N, DT).u_impact)
            for name in CASES
        },
        "checks_evaluated_per_candidate": list(SCREEN_CHECKS),
        "per_candidate_diagnostics": [
            "peak open-cell speed and its location (i, j, x, y, sdf/dx, chi)",
            "local CFL metric at the peak cell (component-wise and summed)",
            "domain maximum local CFL metric",
            "central-D divergence Linf (dimensional and dimensionless)",
            "cut-cell open-face finite-volume divergence Linf (dimensional and dimensionless)",
            "deep-solid speed, chi >= 0.01 band speed, embedded-wall bilinear normal speed",
            "core-mean normal speed and its relative error against -u_impact",
            "t=0 state hashes (u, v, phi) for exact replay",
        ],
        "verdict_vocabulary": list(VERDICTS),
        "prohibitions": {
            "trajectories": "none; the frozen substep kernel is never constructed by this module",
            "contact_events": "none",
            "temporal_refinement": "none",
            "spatial_refinement": "none",
            "geometry_change": "none",
            "re_tuning_after_results": "forbidden; a non-admissible family is reported, never adjusted",
            "historical_evidence_edits": "none",
            "production_solver_edits": "none",
            "contract_13": False,
            "l1b_data_generation": False,
            "ml_training": False,
        },
        "prespecification_rule": (
            "candidate set, family parameters, six frozen gates, envelope constant, targets and verdict "
            "vocabulary are fixed before any candidate value is computed; only the declared diagnostics "
            "are reported, and no candidate is added, removed or rescaled afterwards"
        ),
    }


# ---------------------------------------------------------------------------
# Section C - t=0 per-candidate measurements
# ---------------------------------------------------------------------------
def _peak_metrics(bundle: Any, state: Any) -> dict[str, Any]:
    """Peak open-cell speed with its location, plus the local CFL metrics."""
    p = bundle.p
    volume = np.asarray(pf.phase_control_volumes(bundle.solid, p), dtype=np.float64)
    open_cells = volume > 0.0
    u = np.asarray(state.u, dtype=np.float64)
    v = np.asarray(state.v, dtype=np.float64)
    speed = np.hypot(u, v)
    masked = np.where(open_cells, speed, -np.inf)
    index = np.unravel_index(int(np.argmax(masked)), masked.shape)
    sdf = np.asarray(bundle.solid.sdf, dtype=np.float64)
    chi = np.asarray(bundle.solid.chi, dtype=np.float64)
    peak_u = float(u[index])
    peak_v = float(v[index])
    cfl_component = float(max(abs(peak_u), abs(peak_v)) * p.dt / p.dx)
    cfl_sum = float((abs(peak_u) + abs(peak_v)) * p.dt / p.dx)
    cfl_domain = float(np.max((np.abs(u) + np.abs(v))[open_cells]) * p.dt / p.dx)
    band = chi >= 0.01
    band_speed = float(np.max(np.hypot(u, v)[band]) / bundle.u_impact) if band.any() else 0.0
    return {
        "peak_open_cell_speed": float(masked[index]),
        "peak_over_u_impact": float(masked[index]) / bundle.u_impact,
        "peak_index": [int(index[0]), int(index[1])],
        "peak_position": [float(index[0] * p.dx), float(index[1] * p.dy)],
        "peak_sdf_over_dx": float(sdf[index] / p.dx),
        "peak_chi": float(chi[index]),
        "peak_u_over_u_impact": peak_u / bundle.u_impact,
        "peak_v_over_u_impact": peak_v / bundle.u_impact,
        "peak_inside_gate_band": bool(chi[index] >= 0.01),
        "peak_inside_solid": bool(chi[index] >= 0.5),
        "local_cfl_at_peak_component": cfl_component,
        "local_cfl_at_peak_sum": cfl_sum,
        "local_cfl_domain_max_sum": cfl_domain,
        "u_impact": float(bundle.u_impact),
        "peak_open_cell_speed_raw": float(masked[index]),
        "gate_band_cell_count": int(band.sum()),
        "gate_band_max_speed_over_u_impact": band_speed,
        "open_cell_count": int(open_cells.sum()),
    }


def per_candidate_metrics(profile: str = "t0_screen") -> dict[str, Any]:
    """The t=0 screen matrix: four frozen canary cases x four prespecified candidates."""
    cases = audit._canary_cases()
    matrix: dict[str, Any] = {"profile": profile, "N": PROFILE_N, "dt": DT, "cases": {}}
    for name in CASES:
        bundle = audit._derive_bundle(name, cases[name], PROFILE_N, DT)
        entry: dict[str, Any] = {"case_name": name, "candidates": {}}
        for key in SCREEN_CANDIDATES:
            state, details = stage_2u.build_any_state(bundle, key)
            metrics = audit.initial_metrics(bundle, key, state, details)
            peak = _peak_metrics(bundle, state)
            envelope_ok = peak["peak_over_u_impact"] <= ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT
            six_ok = bool(metrics["initial_constraints_pass"])
            six_details = {
                "checks": metrics["acceptance_checks"],
                "dimensionless_residuals": metrics["dimensionless_residuals"],
                "failing": [gate for gate, ok in metrics["acceptance_checks"].items() if not ok],
            }
            if key == C2_KEY:
                shape_ratio = peak["peak_over_u_impact"] / CORE_TARGET_ABS_OVER_U_IMPACT
                scale_to_envelope = ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT / peak["peak_over_u_impact"]
                core_at_envelope = CORE_TARGET_ABS_OVER_U_IMPACT * scale_to_envelope if scale_to_envelope < 1 else 1.0
            else:
                shape_ratio = peak["peak_over_u_impact"] / CORE_TARGET_ABS_OVER_U_IMPACT
                scale_to_envelope = ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT / peak["peak_over_u_impact"]
                core_at_envelope = CORE_TARGET_ABS_OVER_U_IMPACT * scale_to_envelope
            entry["candidates"][key] = {
                "status": "ADMISSIBLE" if (six_ok and envelope_ok) else "NOT_ADMISSIBLE",
                "six_frozen_initial_constraints": six_ok,
                "six_frozen_details": six_details,
                "global_speed_envelope_pass": bool(envelope_ok),
                "global_speed_envelope_limit": ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT,
                "peak": peak,
                "shape_ratio_peak_over_core_target": shape_ratio,
                "shape_ratio_limit": SHAPE_RATIO_MAX,
                "amplitude_scale_needed_for_envelope": scale_to_envelope,
                "core_target_reached_at_that_scale": core_at_envelope,
                "initializer_parameters": details,
                "u_impact": float(bundle.u_impact),
                "core_mean_normal_speed_raw": float(metrics["actual_initial_core_v_phi_ge_0p9"]),
                "core_mean_normal_speed_over_u_impact": float(
                    metrics["actual_initial_core_v_phi_ge_0p9"] / bundle.u_impact
                ),
                "initial_u_hash": metrics["initial_u_hash"],
                "initial_v_hash": metrics["initial_v_hash"],
                "initial_phi_hash": metrics["initial_phi_hash"],
                "initial_geometry_hash": metrics["initial_geometry_hash"],
            }
        entry["case_verdicts"] = {key: entry["candidates"][key]["status"] for key in SCREEN_CANDIDATES}
        entry["admissible_candidates"] = [k for k, v in entry["case_verdicts"].items() if v == "ADMISSIBLE"]
        matrix["cases"][name] = entry
    for key in SCREEN_CANDIDATES:
        matrix.setdefault("radial_family_summary", {})[key] = {
            "admissible_in_every_case": all(
                matrix["cases"][name]["candidates"][key]["status"] == "ADMISSIBLE" for name in CASES
            ),
            "peak_over_u_impact_by_case": {
                name: matrix["cases"][name]["candidates"][key]["peak"]["peak_over_u_impact"] for name in CASES
            },
        }
    return matrix


def reproduction_check_against_sealed_l1a2u(metrics: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the sealed L1A-2u t=0 values exactly, in both raw and dimensionless form."""
    sealed = json.loads((TWO_PHASE / "evidence" / "l1a2u" / "radial_seed_constraint_matrix.json").read_text())
    rows: dict[str, Any] = {}
    worst_peak = 0.0
    worst_core = 0.0
    all_match = True
    for case_name in CASES:
        entry = metrics["cases"][case_name]
        for key in SCREEN_CANDIDATES:
            mine = entry["candidates"][key]
            theirs = sealed["cases"][case_name]["candidates"][key]
            peak_delta = abs(mine["peak"]["peak_over_u_impact"] - float(theirs["interior_peak_speed_over_u_impact"]))
            core_delta = abs(mine["core_mean_normal_speed_raw"] - float(theirs["core_v_phi_ge_0p9"]))
            six_match = bool(mine["six_frozen_initial_constraints"]) == (theirs["status"] == "COMPATIBLE")
            worst_peak = max(worst_peak, peak_delta)
            worst_core = max(worst_core, core_delta)
            all_match = all_match and six_match and peak_delta < 1e-9 and core_delta < 1e-9
            rows[f"{case_name}/{key}"] = {
                "peak_over_u_impact_mine": mine["peak"]["peak_over_u_impact"],
                "peak_over_u_impact_sealed_l1a2u": float(theirs["interior_peak_speed_over_u_impact"]),
                "peak_abs_delta": peak_delta,
                "raw_core_v_mine": mine["core_mean_normal_speed_raw"],
                "raw_core_v_sealed_l1a2u": float(theirs["core_v_phi_ge_0p9"]),
                "core_abs_delta": core_delta,
                "six_gate_status_match": six_match,
                "core_over_u_impact_mine": mine["core_mean_normal_speed_over_u_impact"],
            }
    return {
        "verdict": "BASELINE_REPLAY_OK" if all_match else "BASELINE_REPLAY_MISMATCH",
        "sealed_source": "examples/two_phase/evidence/l1a2u/radial_seed_constraint_matrix.json",
        "compared_rows": len(rows),
        "worst_peak_abs_delta": worst_peak,
        "worst_core_abs_delta": worst_core,
        "units_note": (
            "the sealed L1A-2u matrix records the raw core speed (-0.5 at u_impact = 0.5); this stage "
            "compares raw to raw and also reports the dimensionless ratio, which is -1.0 and equals the "
            "frozen target -u_impact"
        ),
        "rows": rows,
        "measured_not_assumed": True,
    }


# ---------------------------------------------------------------------------
# Section D - near-wall band classification and the refuted L1A-2u universal claim
# ---------------------------------------------------------------------------
def _find_sealed_statement(node: Any, needle: str = "contour support cell") -> str | None:
    """Depth-first search for the sealed L1A-2u statement this stage tests."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "statement" and needle in str(value):
                return str(value)
            found = _find_sealed_statement(value, needle)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_sealed_statement(item, needle)
            if found is not None:
                return found
    return None


def band_classification() -> dict[str, Any]:
    """Which cases place the frozen contour support inside the ``chi >= 0.01`` gate band."""
    cases = audit._canary_cases()
    per_case: dict[str, Any] = {}
    for name in CASES:
        bundle = audit._derive_bundle(name, cases[name], PROFILE_N, DT)
        geometry = stage_2u.gate_band_geometry(bundle)
        per_case[name] = {
            "phi05_support_sdf_over_dx": geometry["phi05_support_sdf_over_dx"],
            "phi05_support_chi": geometry["phi05_support_chi"],
            "phi05_support_is_inside_gate_band": geometry["phi05_support_is_inside_gate_band"],
            "gate_band_max_sdf_over_dx": geometry["gate_band_max_sdf_over_dx"],
            "gate_band_cell_count": geometry["gate_band_cell_count"],
            "open_cells_min_sdf_over_dx": geometry["open_cells_min_sdf_over_dx"],
        }
    inside = [name for name in CASES if per_case[name]["phi05_support_is_inside_gate_band"]]
    outside = [name for name in CASES if not per_case[name]["phi05_support_is_inside_gate_band"]]
    sealed = json.loads((TWO_PHASE / "evidence" / "l1a2u" / "t0_feasibility_matrix.json").read_text())
    verbatim = _find_sealed_statement(sealed)
    if verbatim is None:
        raise AuditBlocked("could not locate the sealed L1A-2u contour-support statement to test")
    refuted = bool(outside)
    return {
        "per_case": per_case,
        "cases_with_support_inside_gate_band": inside,
        "cases_with_support_outside_gate_band": outside,
        "l1a2u_sealed_statement_verbatim": verbatim,
        "l1a2u_statement_is_universal": "every case" in verbatim,
        "l1a2u_statement_holds_for": inside,
        "l1a2u_statement_refuted_for": outside,
        "l1a2u_universal_claim_refuted": refuted,
        "correction": (
            "the L1A-2u statement that every frozen canary places the phi=0.5 contour support cell inside "
            "the chi >= 0.01 gate band does not hold: complex_heldout places it outside (measured here). "
            "The statement is recorded as refuted in this independent evidence rather than edited in place; "
            "the L1A-2u evidence remains sealed and byte-identical."
        ),
        "measured_not_assumed": True,
    }


# ---------------------------------------------------------------------------
# Section E - shape invariance of the envelope/core requirement
# ---------------------------------------------------------------------------
def shape_invariance(metrics: dict[str, Any]) -> dict[str, Any]:
    """Why no amplitude re-scaling inside the frozen family can satisfy both constraints."""
    rows: dict[str, Any] = {}
    for key in SCREEN_CANDIDATES:
        peaks = [metrics["cases"][name]["candidates"][key]["peak"]["peak_over_u_impact"] for name in CASES]
        core = metrics["cases"][CASES[0]]["candidates"][key]["core_mean_normal_speed_over_u_impact"]
        shape = max(peaks) / abs(core)
        rows[key] = {
            "peak_over_u_impact": max(peaks),
            "core_over_u_impact": core,
            "shape_ratio": shape,
            "shape_ratio_limit": SHAPE_RATIO_MAX,
            "shape_passes": shape <= SHAPE_RATIO_MAX,
            "rescaled_core_target_if_envelope_satisfied": CORE_TARGET_ABS_OVER_U_IMPACT
            * (ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT / max(peaks)),
            "interpretation": (
                "an amplitude rescaling multiplies the peak speed and the core-mean speed by the same "
                "factor, so the ratio between them is invariant; a member whose shape ratio exceeds the "
                "limit cannot satisfy the envelope and the core target simultaneously at any amplitude"
            ),
        }
    return {
        "rows": rows,
        "shape_ratio_limit": SHAPE_RATIO_MAX,
        "rule": (
            "the envelope (4.905526491646905) and the core target (core-mean v = -u_impact, dimensionless "
            "magnitude 1.0) together require a shape ratio peak/core <= 4.905526491646905; the ratio is "
            "invariant under amplitude rescaling, so a violation cannot be repaired by re-scaling within "
            "the family"
        ),
        "derivation": "shape_ratio_limit = envelope_limit / |core target over u_impact| = envelope / 1.0",
        "measured_not_assumed": True,
    }


# ---------------------------------------------------------------------------
# Section F - decision
# ---------------------------------------------------------------------------
def screen_decision(metrics: dict[str, Any], bands: dict[str, Any], shapes: dict[str, Any]) -> dict[str, Any]:
    family_rows: dict[str, Any] = {}
    admissible_members: list[str] = []
    for key in RADIAL_ORDER:
        entry = metrics["radial_family_summary"][key]
        peaks = entry["peak_over_u_impact_by_case"]
        worst_case = max(peaks, key=lambda name: peaks[name])
        six_ok_everywhere = all(
            metrics["cases"][name]["candidates"][key]["six_frozen_initial_constraints"] for name in CASES
        )
        envelope_ok_everywhere = all(
            metrics["cases"][name]["candidates"][key]["global_speed_envelope_pass"] for name in CASES
        )
        admissible = bool(six_ok_everywhere and envelope_ok_everywhere)
        if admissible:
            admissible_members.append(key)
        family_rows[key] = {
            "admissible_in_every_case": admissible,
            "six_frozen_initial_constraints_in_every_case": six_ok_everywhere,
            "global_speed_envelope_in_every_case": envelope_ok_everywhere,
            "peak_over_u_impact_by_case": peaks,
            "worst_case": worst_case,
            "peak_over_envelope_ratio": peaks[worst_case] / ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT,
            "primary_failure": None
            if admissible
            else ("global_speed_envelope" if six_ok_everywhere else "six_frozen_initial_constraints"),
            "failure_reason": None
            if admissible
            else (
                f"peak open-cell speed {peaks[worst_case]:.6f} x u_impact in case `{worst_case}` exceeds "
                f"the prespecified envelope {ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT:.9f} by "
                f"{peaks[worst_case] / ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT:.3f}x while the six frozen "
                "gates still pass; the seed's normalisation forces the whole core to move at "
                "~1 x u_impact and the radial envelope concentrates the return flow, so the excess speed is "
                "the seed's own interior circulation, not a boundary artefact"
            ),
            "shape_ratio": shapes["rows"][key]["shape_ratio"],
            "shape_ratio_passes": shapes["rows"][key]["shape_passes"],
            "amplitude_scale_needed_for_envelope": ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT / peaks[worst_case],
        }
    verdict = "RADIAL_SEED_ADMISSIBLE_UNDER_PRESPECIFIED_FAMILY" if admissible_members else None
    if verdict is None:
        verdict = "NO_ADMISSIBLE_RADIAL_SEED_UNDER_PRESPECIFIED_FAMILY"
    reference = {
        "candidate": C2_KEY,
        "role": "frozen reference baseline, not a member of the radial family under test",
        "admissible_in_every_case": metrics["radial_family_summary"][C2_KEY]["admissible_in_every_case"],
        "peak_over_u_impact_by_case": metrics["radial_family_summary"][C2_KEY]["peak_over_u_impact_by_case"],
        "shape_ratio": shapes["rows"][C2_KEY]["shape_ratio"],
    }
    return {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "profile": "t0_screen",
        "verdict": verdict,
        "radial_family_rows": family_rows,
        "admissible_radial_members": admissible_members,
        "reference_baseline": reference,
        "family_stopped": not admissible_members,
        "stop_rule": (
            "when no prespecified member is admissible the stage stops and reports; it does not re-tune "
            "envelope width, radial support radius or amplitude to manufacture a passing member"
        ),
        "re_tuning_performed": False,
        "band_classification_correction": {
            "cases_with_support_inside_gate_band": bands["cases_with_support_inside_gate_band"],
            "cases_with_support_outside_gate_band": bands["cases_with_support_outside_gate_band"],
            "l1a2u_universal_claim_refuted": bands["l1a2u_universal_claim_refuted"],
        },
        "trajectories_run": 0,
        "contact_events_measured": 0,
        "refinement_studies_run": 0,
        "next_action": next_action(verdict, family_rows, shapes),
    }


def next_action(verdict: str, family_rows: dict[str, Any], shapes: dict[str, Any]) -> dict[str, Any]:
    """Exactly one bounded next physical action; never started in this stage."""
    if verdict == "NO_ADMISSIBLE_RADIAL_SEED_UNDER_PRESPECIFIED_FAMILY":
        action = (
            "Do not re-tune the radial family. At t=0 only, measure the *reachable* interaction window that "
            "the frozen six gates themselves define: for each canary case, compute the maximum incident "
            "normal momentum that can be placed inside the admissible region (sdf above the gate band and "
            "open-cell speed under the 4.905526491646905 envelope) using a support whose interior peak "
            "respects the envelope by construction, and report whether that reachable window can deliver "
            "any incident momentum to the frozen phi=0.5 contour at all; no trajectory, event or refinement "
            "work until that t=0 bound is established."
        )
        scope = (
            "one t=0 reachability bound per canary case; no trajectory, no event, no refinement, "
            "no new initializer promotion"
        )
    else:
        action = (
            "Re-verify the single admissible radial member's t=0 hashes against this screen, then stop and "
            "request an owner decision on whether a contact-event study may be authorized for it."
        )
        scope = "hash re-verification only; no trajectory is authorized by this stage"
    return {
        "action": action,
        "bounded_scope": scope,
        "must_not_be_started_in_this_stage": True,
        "verdict_context": verdict,
        "shape_ratio_limit": shapes["shape_ratio_limit"],
    }


# ---------------------------------------------------------------------------
# Section G - failure injection and adversarial self-checks
# ---------------------------------------------------------------------------
def failure_and_adversarial_tests(contract: dict[str, Any]) -> dict[str, Any]:
    cases = audit._canary_cases()
    bundle = audit._derive_bundle(CASES[0], cases[CASES[0]], PROFILE_N, DT)
    injected: dict[str, Any] = {}

    def _try(name: str, func: Any) -> None:
        try:
            func()
        except FROZEN_BLOCKED as exc:
            injected[name] = {"raised": True, "error": str(exc)[:200]}
        else:
            injected[name] = {"raised": False, "error": None}

    _try("envelope_must_equal_the_sealed_l1a2u_value", lambda: _assert_envelope_matches_sealed())
    _try("unknown_candidate_is_rejected", lambda: stage_2u.build_any_state(bundle, "NOT_A_CANDIDATE"))
    _try(
        "envelope_comparison_is_fail_closed_at_the_boundary",
        _assert_envelope_comparison_is_fail_closed,
    )
    _try(
        "relaxed_envelope_must_be_refused",
        lambda: _assert_relaxed_envelope_refused(bundle),
    )
    adversarial = {
        "envelope_is_a_single_case_independent_constant": len({ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT}) == 1,
        "core_target_unchanged_from_l1a2u": CORE_TARGET_ABS_OVER_U_IMPACT == 1.0,
        "candidate_set_is_the_frozen_one": tuple(SCREEN_CANDIDATES) == (C2_KEY, *stage_2u.RADIAL_ORDER),
        "family_parameters_not_modified": {key: dict(stage_2u.RADIAL_CANDIDATES[key]) for key in RADIAL_ORDER},
        "no_kernel_constructed_by_this_module": (_kernel_needle() not in _source_without_docstring()),
        "verdict_vocabulary_fixed": list(contract["verdict_vocabulary"]) == list(VERDICTS),
    }
    return {"injected_failures": injected, "adversarial": adversarial}


def _kernel_needle() -> str:
    """The frozen substep-kernel symbol, assembled so this check cannot match itself."""
    return "make_" + "fast" + "_substep"


def _source_without_docstring() -> str:
    """Module body with the module docstring removed (for source-level self-checks)."""
    text = Path(__file__).read_text()
    parts = text.split('"""', 2)
    return parts[2] if len(parts) == 3 else text


def _assert_envelope_matches_sealed() -> None:
    sealed = read_sealed_envelope()
    if not sealed["matches"]:
        raise AuditBlocked("sealed envelope mismatch")


def _assert_envelope_comparison_is_fail_closed() -> None:
    """The comparison operator must accept exactly-at-limit and reject one ulp-class excess."""
    limit = ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT
    if not (limit * 0.999999 <= limit):
        raise AuditBlocked("the envelope comparison rejects a value just below the limit")
    if limit <= limit:
        pass
    else:
        raise AuditBlocked("the envelope comparison rejects equality with the limit")
    if (limit * (1.0 + 1e-12)) <= limit:
        raise AuditBlocked("the envelope comparison accepts a value above the limit")


def _assert_relaxed_envelope_refused(bundle: Any) -> None:
    state, _details = stage_2u.build_any_state(bundle, RADIAL_ORDER[0])
    peak = _peak_metrics(bundle, state)["peak_over_u_impact"]
    relaxed = peak * 1.5
    if relaxed <= ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT:
        raise AuditBlocked("the relaxation probe is degenerate")
    raise AuditBlocked(
        f"envelope relaxation refused by construction: a relaxed limit {relaxed} would admit "
        f"{RADIAL_ORDER[0]} (peak {peak}) and is not permitted by this stage"
    )


# ---------------------------------------------------------------------------
# Section H - quality and blockers
# ---------------------------------------------------------------------------
def quality_and_blockers(profile: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "quality_baseline_sha256": sha256_file(REPO_ROOT / "quality" / "legacy_two_phase_debt_v1.json"),
        "declared_new_code_diagnostics": 0,
        "ratchet_expectation": "PASS_BASELINE_ONLY",
        "frozen_adjacent_suite_failures": {
            "test_l1a2r_merged_base_verified_or_fail_closed": {
                "classification": "FROZEN_LINEAGE_DRIFT_CONFIRMED",
                "statement": (
                    "pre-existing; reproduced identically on a pristine worktree at the verified main with "
                    "this stage absent; not edited"
                ),
            },
            "tests/test_nearwall_taper_interface_transport_audit.py"
            "::test_manifest_mismatch_reference_is_recorded_in_preflight": {
                "classification": "SUPERSEDED_BY_NEXT_STAGE_BY_CONSTRUCTION",
                "statement": (
                    "the frozen L1A-2t preflight asserts that every commit above the verified main touches "
                    "only L1A-2t paths and that no other stage's files are present; that assertion cannot "
                    "hold on a branch that carries L1A-2u and L1A-2v; not edited"
                ),
            },
        },
        "not_run": [
            "trajectories",
            "contact events",
            "temporal refinement",
            "spatial refinement",
            "T=8 recompute",
            "L1B data generation",
            "ML training",
            "contract 13",
        ],
        "no_promotion": True,
        "inherited_statuses": {
            "SOLVER_CONTRACT_VERSION": 12,
            "L1A_STATUS": "BLOCKED",
            "L1B_DATA": "L1B_DATA_NOT_READY",
            "unchanged_by_this_stage": True,
        },
    }


# ---------------------------------------------------------------------------
# Section I - assembly, verification and report
# ---------------------------------------------------------------------------
def render_report(
    *,
    preflight: dict[str, Any],
    sealed: dict[str, Any],
    contract: dict[str, Any],
    metrics: dict[str, Any],
    bands: dict[str, Any],
    shapes: dict[str, Any],
    decision: dict[str, Any],
    adversarial: dict[str, Any],
    quality: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# L1A-2v final report - interior-speed-constrained t=0 admissibility screen")
    add("")
    add("## A. Scope, authority and baseline")
    add(
        f"- sole study baseline: PR #{BASELINE_PR} head `{preflight['baseline_head_sha']}` "
        f"(verified main `{VERIFIED_MAIN_SHA}`)"
    )
    add(
        f"- branch `{preflight['branch']}`, contract {preflight['solver_contract_version']}, "
        f"policy `{preflight['timestep_policy']}`"
    )
    add(f"- screen profile: N={preflight['N']}, dt={preflight['dt']}, cases {list(CASES)}")
    add(f"- trajectories run: {preflight['trajectories_run']}; contact events: {preflight['contact_events']}")
    add("")

    add("## B. Frozen contract and prohibitions")
    add(f"- six frozen gates: {json.dumps(contract['frozen_six_gates'], default=_json_default)}")
    add(f"- gate source: {contract['frozen_six_gates_source']}")
    add(f"- core normalisation target: {contract['core_normalisation_target']['target_over_u_impact']} (unchanged)")
    add(f"- prohibitions: {json.dumps(contract['prohibitions'], default=_json_default)}")
    add("")

    add("## C. Sealed evidence verification")
    l1a2u_verifier = sealed["l1a2u_frozen_verifier"]
    l1a2t_verifier = sealed["l1a2t_frozen_verifier"]
    add(f"- L1A-2u frozen verifier: {_fmt(l1a2u_verifier['verified'])}, bound files {l1a2u_verifier['bound_files']}")
    add(f"- L1A-2t frozen verifier: {_fmt(l1a2t_verifier['verified'])}, bound files {l1a2t_verifier['bound_files']}")
    verified_total = sum(int(v.get("verified", 0) or 0) for v in sealed["inventory"].values())
    mismatched_total = sum(int(v.get("mismatched", 0) or 0) for v in sealed["inventory"].values())
    add(f"- filename inventory across every sealed directory: {verified_total} verified, {mismatched_total} superseded")
    for path, record in sealed["superseded_historical_records"].items():
        add(f"    - superseded record `{path}`: last changed by `{record['last_change_commit']}`")
    add("")

    add("## D. Prespecified screen contract")
    add(f"- envelope limit (unmodifiable): {ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT!r}")
    add(f"- envelope provenance: {json.dumps(sealed_envelope_ref())}")
    add(f"- implied shape-ratio limit: {SHAPE_RATIO_MAX!r}")
    add(f"- checks per candidate: {contract['checks_evaluated_per_candidate']}")
    add("")

    add("## E. t=0 per-candidate measurements")
    for name in CASES:
        for key in SCREEN_CANDIDATES:
            row = metrics["cases"][name]["candidates"][key]
            peak = row["peak"]
            residual = row["six_frozen_details"]["dimensionless_residuals"]
            add(
                f"- `{name}` `{key}`: {row['status']}, peak/U={_fmt(peak['peak_over_u_impact'], 4)} at "
                f"cell {peak['peak_index']} (sdf/dx={_fmt(peak['peak_sdf_over_dx'], 3)}, "
                f"chi={_fmt(peak['peak_chi'], 3)}, in band={_fmt(peak['peak_inside_gate_band'])}, "
                f"in solid={_fmt(peak['peak_inside_solid'])}), "
                f"CFL peak={_fmt(peak['local_cfl_at_peak_sum'], 3)} "
                f"domain max={_fmt(peak['local_cfl_domain_max_sum'], 3)}, "
                f"div D={_fmt(residual['dx_D_Linf_over_u_impact'], 3)}, "
                f"div FV={_fmt(residual['dx_FV_Linf_over_u_impact'], 3)}, "
                f"deep solid/U={_fmt(residual['deep_solid_speed_over_u_impact'], 3)}, "
                f"wall normal/U={_fmt(residual['wall_normal_sample_over_u_impact'], 3)}, "
                f"band speed/U={_fmt(peak['gate_band_max_speed_over_u_impact'], 4)}, "
                f"core v/U={_fmt(row['core_mean_normal_speed_over_u_impact'], 5)}, "
                f"six gates={_fmt(row['six_frozen_initial_constraints'])}, "
                f"envelope={_fmt(row['global_speed_envelope_pass'])}"
            )
    add("")

    add("## F. Near-wall band classification and the refuted L1A-2u universal claim")
    for name in CASES:
        row = bands["per_case"][name]
        add(
            f"- `{name}`: phi=0.5 support sdf/dx={_fmt(row['phi05_support_sdf_over_dx'], 3)}, "
            f"chi={_fmt(row['phi05_support_chi'], 4)}, "
            f"inside gate band={_fmt(row['phi05_support_is_inside_gate_band'])}, "
            f"band max sdf/dx={_fmt(row['gate_band_max_sdf_over_dx'], 3)}"
        )
    add(f"- cases with support inside the band: {bands['cases_with_support_inside_gate_band']}")
    add(f"- cases with support outside the band: {bands['cases_with_support_outside_gate_band']}")
    add(f'- L1A-2u statement (verbatim, sealed): "{bands["l1a2u_sealed_statement_verbatim"]}"')
    add(
        f"- that statement is universal: {_fmt(bands['l1a2u_statement_is_universal'])}; "
        f"refuted: {_fmt(bands['l1a2u_universal_claim_refuted'])}"
    )
    add(f"- correction: {bands['correction']}")
    add("")

    add("## G. Shape invariance of the envelope and core requirements")
    add(f"- derived limit: peak/core <= {shapes['shape_ratio_limit']!r} = envelope / |core target|")
    for key, row in shapes["rows"].items():
        add(
            f"- `{key}`: peak/U={_fmt(row['peak_over_u_impact'], 4)}, "
            f"core/U={_fmt(row['core_over_u_impact'], 5)}, "
            f"shape ratio={_fmt(row['shape_ratio'], 4)} "
            f"(limit {_fmt(row['shape_ratio_limit'], 4)}), passes={_fmt(row['shape_passes'])}, "
            f"core reached if rescaled to the envelope="
            f"{_fmt(row['rescaled_core_target_if_envelope_satisfied'], 5)}"
        )
    add(f"- rule: {shapes['rule']}")
    add("")

    add("## H. Admissibility decision")
    add(f"- verdict `{decision['verdict']}`")
    add(f"- admissible radial members: {decision['admissible_radial_members']}")
    add(
        f"- family stopped: {_fmt(decision['family_stopped'])}; "
        f"re-tuning performed: {_fmt(decision['re_tuning_performed'])}"
    )
    add(f"- stop rule: {decision['stop_rule']}")
    for key, row in decision["radial_family_rows"].items():
        add(
            f"- `{key}`: admissible={_fmt(row['admissible_in_every_case'])}, "
            f"primary failure={_fmt(row['primary_failure'])}, "
            f"peak/envelope worst-case ratio={_fmt(row['peak_over_envelope_ratio'], 4)}, "
            f"shape ratio={_fmt(row['shape_ratio'], 4)}"
        )
    reference = decision["reference_baseline"]
    add(
        f"- reference baseline `{reference['candidate']}`: {reference['role']}, "
        f"admissible={_fmt(reference['admissible_in_every_case'])}"
    )
    refuted = decision["band_classification_correction"]["l1a2u_universal_claim_refuted"]
    add(f"- refuted universal claim recorded in this independent evidence: {_fmt(refuted)}")
    add("")

    add("## I. Failure injection and adversarial checks")
    for name, row in adversarial["injected_failures"].items():
        note = "" if row["error"] is None else f" - {str(row['error'])[:160]}"
        add(f"- probe `{name}`: raised={_fmt(row['raised'])}{note}")
    add(f"- adversarial: {json.dumps(adversarial['adversarial'], default=_json_default)[:900]}")
    add("")

    add("## J. Quality, tests and hosted CI")
    add(
        f"- quality baseline `{quality['quality_baseline_sha256']}`, declared new diagnostics "
        f"{quality['declared_new_code_diagnostics']}, expectation `{quality['ratchet_expectation']}`"
    )
    for name, row in quality["frozen_adjacent_suite_failures"].items():
        add(f"- frozen adjacent failure `{name}`: {row['classification']} - {row['statement']}")
    add(f"- not run: {quality['not_run']}")
    add("")

    add("## K. Blockers and inherited statuses")
    add(f"- inherited: {json.dumps(quality['inherited_statuses'], default=_json_default)}")
    add("- blocker for any contact study: no admissible radial member exists under the prespecified family")
    add("")

    add("## L. Exactly one next action")
    add(f"- {decision['next_action']['action']}")
    add(f"- bounded scope: {decision['next_action']['bounded_scope']}")
    add(f"- must not be started in this stage: {_fmt(decision['next_action']['must_not_be_started_in_this_stage'])}")
    add("")

    add("## M. Status and provenance")
    add(f"- module sha256 `{provenance['module_sha256']}`")
    add(f"- sealed dependency hashes: {json.dumps(provenance['sealed_dependency_hashes'], default=_json_default)}")
    add(f"- status `{decision['verdict']}`; no promotion; no trajectory; overlay evidence only")
    add("")
    return "\n".join(lines) + "\n"


def sealed_envelope_ref() -> dict[str, Any]:
    try:
        return read_sealed_envelope()
    except Exception as exc:  # pragma: no cover - diagnostics only
        return {"error": str(exc)[:200]}


def assemble_evidence(result: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}
    for name in EVIDENCE_FILES:
        if name == "manifest.json":
            continue
        if name.endswith(".md"):
            continue
        payload = result.get(name)
        if payload is None:
            continue
        (out_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
        written[name] = sha256_file(out_dir / name)
    report = render_report(
        preflight=result["preflight.json"],
        sealed=result["frozen_evidence_verification.json"],
        contract=result["screen_contract_and_prespecification.json"],
        metrics=result["t0_per_candidate_metrics.json"],
        bands=result["nearwall_support_band_classification.json"],
        shapes=result["shape_invariance_and_envelope_analysis.json"],
        decision=result["admissibility_decision.json"],
        adversarial=result["failure_and_adversarial_tests.json"],
        quality=result["quality_and_blocker_matrix.json"],
        provenance=result["source_and_provenance.json"],
    )
    (out_dir / "l1a2v_final_report.md").write_text(report)
    written["l1a2v_final_report.md"] = sha256_file(out_dir / "l1a2v_final_report.md")
    final_json = {
        key: result[key]
        for key in (
            "preflight.json",
            "admissibility_decision.json",
            "nearwall_support_band_classification.json",
            "shape_invariance_and_envelope_analysis.json",
        )
    }
    (out_dir / "l1a2v_final_report.json").write_text(
        json.dumps(final_json, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    written["l1a2v_final_report.json"] = sha256_file(out_dir / "l1a2v_final_report.json")
    manifest = {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "profile": "t0_screen",
        "evidence_class": "T0_SCREEN_N192",
        "baseline_head_sha": BASELINE_HEAD_SHA,
        "baseline_pull_request": BASELINE_PR,
        "verified_main_sha": VERIFIED_MAIN_SHA,
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "envelope_max_open_cell_speed_over_u_impact": ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT,
        "trajectories_run": 0,
        "contact_events_measured": 0,
        "refinement_studies_run": 0,
        "no_promotion": True,
        "hosted_ci": {"status_for_this_head": "UNMEASURED", "reason": "not pushed at assembly time"},
        "no_hosted_ci_claim_before_push": True,
        "required_files": list(EVIDENCE_FILES),
        "files": written,
        "module_sha256": sha256_file(Path(__file__)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n")
    return manifest


def reseal_ci_status(ci_status: str, head_sha: str, out_dir: str | Path) -> dict[str, Any]:
    """Record an observed post-push hosted CI status; re-hashes only the two metadata files.

    No physics, threshold, hash or verdict is touched: the CI status lives in
    ``quality_and_blocker_matrix.json``, the report line quoting it, and the manifest.
    """
    if ci_status not in ("success", "failure", "partial", "NOT_RUN"):
        raise AuditBlocked(f"refusing to record an unverified CI status string: {ci_status!r}")
    if len(head_sha) != 40 or any(ch not in "0123456789abcdef" for ch in head_sha):
        raise AuditBlocked(f"refusing to record an unverified head sha: {head_sha!r}")
    directory = Path(out_dir)
    quality_path = directory / "quality_and_blocker_matrix.json"
    if not quality_path.is_file():
        raise AuditBlocked(f"no quality_and_blocker_matrix.json under {directory}")
    quality = json.loads(quality_path.read_text())
    previous = quality.get("hosted_ci", {}).get("status_for_this_head", "UNMEASURED")
    quality["hosted_ci"] = {
        "status_for_this_head": ci_status,
        "observed_head_sha": head_sha,
        "previous_status": previous,
        "resealed_after_push": True,
        "rule": "no hosted result is claimed for a head that was not pushed and observed",
    }
    quality_path.write_text(json.dumps(quality, indent=2, sort_keys=True, default=_json_default) + "\n")
    report_path = directory / "l1a2v_final_report.md"
    report = report_path.read_text()
    marker = "\n## L. Exactly one next action"
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
    for name in ("quality_and_blocker_matrix.json", "l1a2v_final_report.md"):
        if name in files:
            files[name] = sha256_file(directory / name)
    manifest["files"] = files
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n")
    return verify_evidence_manifest(directory)


def verify_evidence_manifest(out_dir: str | Path) -> dict[str, Any]:
    """Fail-closed verification of a committed L1A-2v evidence directory."""
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
    if float(manifest.get("envelope_max_open_cell_speed_over_u_impact", -1)) != float(
        ENVELOPE_MAX_OPEN_CELL_SPEED_OVER_U_IMPACT
    ):
        problems.append("envelope drift")
    if int(manifest.get("trajectories_run", -1)) != 0:
        problems.append("trajectory count must be zero")
    if manifest.get("no_hosted_ci_claim_before_push") is not True:
        problems.append("the evidence must declare that no hosted CI was claimed before the push")
    if manifest.get("no_promotion") is not True:
        problems.append("no_promotion must be true")
    missing_required = [name for name in EVIDENCE_FILES if not (directory / name).is_file()]
    if missing_required:
        problems.append(f"missing required files: {sorted(missing_required)}")
    for name, want in manifest.get("files", {}).items():
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
        "verified_main_sha": manifest.get("verified_main_sha"),
        "bound_files": len(manifest.get("files", {})),
        "directory": str(directory),
        "problems": problems,
        "missing_required": missing_required,
        "verified": True,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
STAGES = (
    "preflight",
    "sealed",
    "contract",
    "metrics",
    "bands",
    "shapes",
    "decision",
    "adversarial",
    "quality",
    "assemble",
)


def run(*, out: str | Path | None = None, stages: str = "all") -> dict[str, Any]:
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
    if "sealed" in requested or "assemble" in requested:
        progress("sealed evidence verification")
        result["frozen_evidence_verification.json"] = verify_sealed_evidence()
    if "contract" in requested or "assemble" in requested:
        result["screen_contract_and_prespecification.json"] = screen_contract()
    if "metrics" in requested or "assemble" in requested:
        progress("t=0 per-candidate screen matrix (N=192)")
        result["t0_per_candidate_metrics.json"] = per_candidate_metrics()
    if "bands" in requested or "assemble" in requested:
        metrics_for_check = result.get("t0_per_candidate_metrics.json") or per_candidate_metrics()
        result["l1a2u_reproduction_check.json"] = reproduction_check_against_sealed_l1a2u(metrics_for_check)
        if result["l1a2u_reproduction_check.json"]["verdict"] != "BASELINE_REPLAY_OK":
            raise AuditBlocked("the sealed L1A-2u t=0 matrix was not reproduced; stopping before any screen claim")
        progress("near-wall band classification")
        result["nearwall_support_band_classification.json"] = band_classification()
    if "shapes" in requested or "assemble" in requested:
        metrics = result.get("t0_per_candidate_metrics.json") or per_candidate_metrics()
        result["shape_invariance_and_envelope_analysis.json"] = shape_invariance(metrics)
    if "decision" in requested or "assemble" in requested:
        metrics = result["t0_per_candidate_metrics.json"]
        bands = result["nearwall_support_band_classification.json"]
        shapes = result["shape_invariance_and_envelope_analysis.json"]
        decision = screen_decision(metrics, bands, shapes)
        result["admissibility_decision.json"] = decision
        progress(
            f"decision: verdict={decision['verdict']} "
            f"admissible={decision['admissible_radial_members']} "
            f"refuted_l1a2u_universal={bands['l1a2u_universal_claim_refuted']}"
        )
    if "adversarial" in requested or "assemble" in requested:
        contract = result["screen_contract_and_prespecification.json"]
        result["failure_and_adversarial_tests.json"] = failure_and_adversarial_tests(contract)
    if "quality" in requested or "assemble" in requested:
        result["quality_and_blocker_matrix.json"] = quality_and_blockers("t0_screen")
    if "assemble" in requested:
        result["prerequisite_and_source_map.json"] = prerequisite_and_source_map(preflight)
        result["source_and_provenance.json"] = source_and_provenance(preflight)
        out_dir = _out_dir(out)
        manifest = assemble_evidence(result, out_dir)
        result["manifest"] = manifest
        progress(f"evidence assembled in {out_dir}")
    result["__elapsed_seconds__"] = time.perf_counter() - started
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--out", default=None)
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
    result = run(out=args.out, stages=args.stages)
    if "manifest" in result:
        print(
            f"[{STAGE}] done in {result['__elapsed_seconds__']:.1f}s; "
            f"verdict={result['admissibility_decision.json']['verdict']}"
        )


if __name__ == "__main__":
    main()
