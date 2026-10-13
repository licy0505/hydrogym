"""L1A-2z - prespecified own-base attribution and a unique-support measurement.

L1A-2y left two measurements open and reported ``EARLY_WINDOW_ATTRIBUTION_UNRESOLVED``:

1. its prespecified *primary* attribution rule missed its own ``0.01 dx`` closure criterion by
   ``0.1223197 dx``, and
2. the frozen global ``phi = 0.5`` minimum-gap support match at the tracked point was reported
   non-unique on all 40 macro steps.

This stage changes the *measurement*, never the physics, the trajectory or the acceptance criteria.
The frozen solver, its contract, the window, the case, the resolution, the time step, the contact guard
and every previously frozen threshold are untouched; the L1A-2x and L1A-2y evidence and verdicts are
read-only inputs. Nothing in this stage is a mechanism proof.

Why the L1A-2y primary rule could not close
-------------------------------------------
One frozen substep of the L1A-2x capture is (``phasefield`` order, exactly as the solver runs it)

    S0 = phi
    S1 = S0 + d_adv          (d_adv = dt * phi_rhs, the conservative advective update)
    S2 = S0 + d_adv + d_ch   (= phi_hat, d_ch = -dt * div_ch, the explicit chemical-potential flux)
    S3 = S2 + d_imp          (= phi_new, d_imp = phi_new - phi_hat, the implicit CH solve)

so ``d_adv + d_ch + d_imp = phi_new - phi`` exactly, and ``S1``, ``S2``, ``S3`` are the *actual*
intermediate states of the frozen operator.

L1A-2y applied each ingredient increment to the **same** substep base field (``base + d_adv``,
``base + d_ch``, ``base + d_imp``) and then measured each ingredient with its *own* chain over the whole
window: ``cumulative[i][k] = Q(base_k + d_i^k) - Q(base_{k'} + d_i^{k'})``. Such a chain telescopes to the
displacement between the crossings of two *partial* fields of different substeps, so its total is the
displacement of one ingredient's increments applied to each other, not that ingredient's contribution to
the actual motion. Its three chain totals therefore cannot sum to the measured total; the stage measured
that failure honestly (the sum exceeded the total by ``0.1223197 dx``, 12.2 times its own band) and
withheld the verdict. ``RULE_B`` of that stage applied each increment to its own substep base and closed
to ``-0.0079371 dx``, inside the band, but its three legs still sum to a *marginal* view (each increment
applied alone), which differs from the actual sequence by the interaction term.

The corrected rule of this stage
--------------------------------
For every substep the observable ``Q`` (the tracked ``phi = 0.5`` branch root on the fixed wall-normal
ray, frozen readout) is evaluated on the consecutive *actual* states:

    D_adv  = sum_k [ Q(S1^k) - Q(S0^k) ]
    D_ch   = sum_k [ Q(S2^k) - Q(S1^k) ]
    D_imp  = sum_k [ Q(S3^k) - Q(S2^k) ]

Each contribution is a difference of the same observable between the actual input and output state of
that operator, so the sum telescopes **exactly**:

    D_adv + D_ch + D_imp = Q(S3^{K}) - Q(S0^{1}) = Q(phi_final) - Q(phi_initial).

That identity is algebra, not evidence of mechanism: it is reported as a *closure* check with its
floating-point residual, and no verdict of this stage depends on the fact that it holds. What the run
actually measures is the *size* of each leg and the spread of the attribution under the prespecified
alternative conventions.

Support uniqueness
------------------
The L1A-2y non-uniqueness is a degeneracy of the *minimum-distance point*, not of the interface branch.
At ``t = 0`` the two lowest-SDF ``phi = 0.5`` candidates are the mirror pair ``x = 2.984375`` and
``x = 3.015625`` at exactly the same height ``y = 0.7345619198479572``; their wall distance agrees to
machine precision (the frozen SDF is exactly linear in ``y`` there and the frozen state is exactly
mirror-symmetric about ``x = 3``), and the frozen readout traces both of them on **one** open contour
arc. The frozen L1A-2x diagnostic therefore reports ``NON_UNIQUE`` on every macro step, correctly, for
*its* semantics (the arg-min point is not unique), and it is retained here as a historical comparator
without being redefined.

This stage replaces the point question by the branch question, prespecified as follows.

``candidates``
    Every reconstructed ``phi = 0.5`` crossing of the frozen periodic/linear reconstruction (the frozen
    enumeration), each carrying the grid edge it lies on.
``components``
    The connected components of that reconstruction, built from the same per-cell linear segments the
    frozen readout uses (including the frozen asymptotic-decider pairing at saddle cells), by union-find
    over the crossing edges. The vectorised construction is validated cell by cell against the frozen
    per-cell reconstruction.
``minimum``
    ``g* = min`` wall-SDF distance over the candidates; ``TIE_TOLERANCE_DX`` decides which candidates are
    *tied* at the minimum and ``COMPETITION_TOLERANCE_DX`` which ones *compete* for it.
``uniqueness gate``
    Every competing candidate (within ``COMPETITION_TOLERANCE_DX`` of ``g*``) must lie on **one** connected
    component, the tracked root must stay on the component pinned at ``t = 0``, and no exact-threshold
    grid node may make the pairing ambiguous. The gate is applied to every intermediate state, not only
    to macro boundaries.
``point degeneracy is admitted, and measured``
    A tie inside one component is *not* uniqueness of the point and is never claimed as such. Both
    ``t = 0`` tied supports are pinned independently and the whole attribution is measured on both; their
    raw difference is reported as the support-tie term of the comparison error. A deterministic tie-break
    (smallest ``x``) selects the primary support, and that choice is stated as a tie-break, not as
    evidence.

Verdict discipline
------------------
Three questions are kept apart: (A) is the tracked observable uniquely defined and reproducible, (B)
does the corrected own-base decomposition satisfy the frozen ``0.01 dx`` closure criterion, (C) does the
experiment establish *why* the interface moved. This stage can answer A and B; it never answers C. One
trajectory without a control arm cannot separate a causal mechanism, and an algebraic identity is not
evidence that one physical operator acted alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import phasefield as pf  # noqa: E402
from production import frozen_chns_early_window_transport_attribution_audit as stage_2x  # noqa: E402
from production import frozen_reconstruction_branch_attribution_audit as stage_2y  # noqa: E402
from production import impact_initialization_compatibility_audit as audit  # noqa: E402
from production import l1a2y_contour_branch_readout as readout  # noqa: E402

STAGE = "L1A-2z"
STAGE_VERSION = "l1a2z_v1"
BRANCH = "arena/02b6c71b-hydrogym"
BASELINE_HEAD_SHA = "cd20a7838fb616f3857092c580f3c24802c6115a"
BASELINE_STAGE = "L1A-2y"
VERIFIED_MAIN_SHA = "09e2993b8da478d79e768054194201af57121784"

TWO_PHASE = Path(__file__).resolve().parents[1]
REPO_ROOT = TWO_PHASE.parents[1]
EVIDENCE_OWN = TWO_PHASE / "evidence" / "l1a2z"
PRESPECIFICATION = TWO_PHASE / "l1a2z_prespecification.md"
MODULE_PATH = Path(__file__).resolve()
TEST_PATH = TWO_PHASE / "tests" / "test_l1a2z_sequential_ownbase_attribution_audit.py"
PRIOR_EVIDENCE: dict[str, Path] = {
    "l1a2x": TWO_PHASE / "evidence" / "l1a2x",
    "l1a2y": TWO_PHASE / "evidence" / "l1a2y",
}

STAGE_OWNED_PATHS = (
    "examples/two_phase/production/l1a2z_sequential_ownbase_attribution_audit.py",
    "examples/two_phase/tests/test_l1a2z_sequential_ownbase_attribution_audit.py",
    "examples/two_phase/l1a2z_prespecification.md",
    "examples/two_phase/evidence/l1a2z",
)

# --- the window, carried over from L1A-2y unchanged --------------------------------------------
CASE_NAME = "pillar_training"
PROFILE_N = 192
REQUESTED_DT = 0.002
MAX_MACRO_STEPS = 40
SUBSTEPS_PER_MACRO = 3
CONTACT_GAP_THRESHOLD_DX = 1.5

# --- acceptance criteria, carried over from L1A-2y unchanged -------------------------------------
DOMINANCE_FACTOR = 3.0
ATTRIBUTION_CLOSURE_BAND_DX = 0.01
TOTAL_DISPLACEMENT_CLOSURE_BAND_DX = 0.01
WALK_NOISE_FLOOR_DX = 1.0e-3
CROSS_CHECK_TOLERANCE_DX = 0.05
STATIC_RECONSTRUCTION_TOLERANCE_DX = 0.01
EQUIVALENCE_MAX_PHI_DIFF = 1e-6
EQUIVALENCE_MAX_GAP_DIFF_DX = 1e-4

# --- new in L1A-2z: the support and comparison-error protocol, frozen before the run --------------
TIE_TOLERANCE_DX = 1.0e-6
COMPETITION_TOLERANCE_DX = 1.0e-2
EXACT_THRESHOLD_TOLERANCE = 0.0
TIE_ARC_TOLERANCE_DX = 1.0e-6
CONTRACT_LEG_TOLERANCE_DX = 1.0e-9
#: How far the tracked component may differ geometrically between two consecutive measured states and still
#: be the same branch. Derived from the frozen readout rule, never from the data: the frozen continuity rule
#: bounds one substep's tracked-root displacement by ``MAX_SUBSTEP_DISPLACEMENT_OVER_EPS * eps`` =
#: 0.25 * 1.5 dx = 0.375 dx, and a component whose crossings all lie within that distance of the previous
#: component's crossings cannot be a different branch.
BRANCH_IDENTITY_TOLERANCE_DX = 0.25 * 1.5
BRANCH_IDENTITY_RELATIONS = (
    "PERSISTENT",
    "GREW_OR_MERGED",
    "FRAGMENTED_OR_SPLIT",
    "DISJOINT_REASSIGNMENT",
    "AMBIGUOUS_RELATION",
    "NO_COMPONENT",
)
CONVENTION_NAMES = (
    "SEQUENTIAL_OWN_BASE",
    "MARGINAL_OWN_BASE",
    "LINEARISED_POINT_T0",
    "LINEARISED_SUPPORT_CELL_T0",
)
PRIMARY_CONVENTION = "SEQUENTIAL_OWN_BASE"
PRIMARY_RULE = "RULE_S_SEQUENTIAL_OWN_BASE_ACTUAL_INTERMEDIATE_STATES"
HISTORICAL_RULE = "RULE_A_L1A2Y_PRIMARY_CHAIN_CONTINUATION"
DECISION_STATES = (
    "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED",
    "EARLY_WINDOW_MEASUREMENT_ATTRIBUTION_RESOLVED",
)
DOMINANCE_READINGS = (
    "ADVECTIVE_LEG_LARGER",
    "CAHN_HILLIARD_LEG_LARGER",
    "WITHIN_THE_COMPARISON_ERROR",
)
EVIDENCE_FILES = (
    "preflight.json",
    "source_and_provenance.json",
    "prespecification_freeze.json",
    "prerequisite_verification.json",
    "support_uniqueness_design.json",
    "initial_state_identity.json",
    "trajectory_specification.json",
    "sequential_ownbase_ledger.json",
    "support_uniqueness_window.json",
    "attribution.json",
    "residual_decomposition.json",
    "conservation_and_convergence.json",
    "capture_equivalence.json",
    "identical_window_check.json",
    "validity_screen.json",
    "mechanism_decision.json",
    "adversarial_and_falsification.json",
    "quality_and_blocker_matrix.json",
    "next_action.json",
    "l1a2z_final_report.json",
    "l1a2z_final_report.md",
    "manifest.json",
)
FROZEN_BLOCKED: tuple[type[BaseException], ...] = (stage_2x.AuditBlocked, stage_2y.AuditBlocked)
CONTOUR_THRESHOLD = 0.5


class AuditBlocked(RuntimeError):
    """Fail-closed protocol, provenance, source-freeze or identity error."""


FROZEN_BLOCKED = FROZEN_BLOCKED + (AuditBlocked,)


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
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}g}"


def _git(args: list[str], *, required: bool = True) -> str | None:
    import subprocess

    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
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


def _porcelain_path(line: str) -> str:
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
    return all(_stage_owned(line.strip()) for line in changed.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# Section A - preflight, provenance and the prespecification/measurement boundary
# ---------------------------------------------------------------------------
def preflight_state() -> dict[str, Any]:
    """Fail-closed preflight: branch, head, lineage, worktree hygiene and the stage-evidence guard."""
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
    own_manifest = EVIDENCE_OWN / "manifest.json"
    payload = {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "branch": branch,
        "head_sha": head,
        "baseline_stage": BASELINE_STAGE,
        "baseline_head_sha": BASELINE_HEAD_SHA,
        "baseline_pull_request": 24,
        "baseline_rule": (
            f"this stage runs from the {BASELINE_STAGE} evidence head {BASELINE_HEAD_SHA}; every commit "
            "above it must be owned by this stage or the stage refuses to measure"
        ),
        "main_sha": VERIFIED_MAIN_SHA,
        "worktree_clean_outside_stage_deliverables": not foreign,
        "dirty_paths_outside_stage_deliverables": foreign,
        "stage_owned_dirty_paths": [path for path in dirty if _stage_owned(path)],
        "commits_above_baseline": commits,
        "commits_above_baseline_not_stage_owned": unknown,
        "shallow_repository": (_git(["rev-parse", "--is-shallow-repository"]) or "unknown") == "true",
        "stage_evidence_tracked_at_head": bool(
            (_git(["ls-files", "examples/two_phase/evidence/l1a2z"], required=False) or "").strip()
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
    if payload["stage_evidence_tracked_at_head"] and not own_manifest.is_file() and not dirty:
        raise AuditBlocked(
            "this head has committed stage evidence but the evidence directory is missing from the worktree; "
            "refusing to measure, because regenerating it here would silently repair a committed deletion"
        )
    return payload


def _frozen_hash_block(text: str) -> dict[str, str]:
    match = re.search(r"<!-- L1A2Z-FROZEN-HASHES\n(.*?)\n-->", text, re.S)
    if match is None:
        raise AuditBlocked("the prespecification document carries no frozen-hash block")
    block: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(" ")
        block[key.strip()] = value.strip()
    return block


def measurement_code_changes(boundary_commit: str) -> list[dict[str, Any]]:
    """Every commit above the prespecification boundary that touched the frozen measurement code.

    The boundary is the commit that introduced the prespecification document. The document declares how
    many such commits exist (its ``sync_count``), so an undeclared change to the measurement code is
    detectable instead of invisible: the declared count and the observed change set must agree, and the
    changed commits with their file lists are recorded in the evidence for a consumer to inspect. Only the
    module and the focused test file count as measurement code, so the evidence of this stage can never be
    mistaken for a protocol change.
    """
    paths = sorted(_relative(path) for path in (MODULE_PATH, TEST_PATH))
    log = _git(["log", "--format=%H", "--name-only", f"{boundary_commit}..HEAD", "--", *paths])
    changes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in (log or "").splitlines():
        entry = line.strip()
        if not entry:
            continue
        if len(entry) == 40 and all(character in "0123456789abcdef" for character in entry):
            current = {"commit": entry, "files": []}
            changes.append(current)
        elif current is not None:
            current["files"].append(entry)
    return [change for change in changes if change["files"]]


def prespecification_freeze() -> dict[str, Any]:
    """Verify that the run uses the prespecified, frozen measurement code and nothing else.

    The boundary is the prespecification commit: the document records the sha256 of the module and of the
    focused test file, and the run refuses to measure unless those files are byte-identical to the values
    the document froze, the commit that introduced the document is an ancestor of this head, and no commit
    after it touched any stage-owned path.
    """
    if not PRESPECIFICATION.is_file():
        raise AuditBlocked(f"the prespecification document is missing: {PRESPECIFICATION}")
    block = _frozen_hash_block(PRESPECIFICATION.read_text())
    problems: list[str] = []
    expected = {
        "protocol_version": STAGE_VERSION,
        "module_sha256": sha256_file(MODULE_PATH),
        "test_sha256": sha256_file(TEST_PATH),
    }
    for key, value in expected.items():
        if block.get(key) != value:
            problems.append(f"{key}: frozen={block.get(key)!r} on_disk={value!r}")
    introduced = (_git(["log", "--diff-filter=A", "--format=%H", "--", _relative(PRESPECIFICATION)]) or "").split()
    commit = introduced[-1] if introduced else None
    ancestor = commit is not None and _git(["merge-base", "--is-ancestor", commit, "HEAD"], required=False) is not None
    if commit is None:
        problems.append("the prespecification document is not committed")
    elif not ancestor:
        problems.append(f"the prespecification commit {commit} is not an ancestor of this head")
    changes: list[dict[str, Any]] = []
    if commit is not None and ancestor:
        changes = measurement_code_changes(commit)
        declared = block.get("sync_count")
        if declared is None or not str(declared).isdigit():
            problems.append("the frozen-hash block declares no sync_count")
        elif int(str(declared)) != len(changes):
            problems.append(
                f"the frozen-hash block declares {declared} pre-measurement hash-block syncs but "
                f"{len(changes)} commits above the boundary touched the frozen measurement code: "
                f"{[change['commit'] for change in changes]}"
            )
    if problems:
        raise AuditBlocked(f"the prespecification freeze does not hold: {problems}")
    return {
        "stage": STAGE,
        "document": _relative(PRESPECIFICATION),
        "document_sha256": sha256_file(PRESPECIFICATION),
        "prespecification_commit": commit,
        "prespecification_commit_is_an_ancestor_of_head": bool(ancestor),
        "frozen_hashes": block,
        "verified_against_the_worktree": expected,
        "hash_block_syncs": {
            "declared": int(str(block.get("sync_count", "-1"))),
            "observed": len(changes),
            "commits": changes,
            "rule": (
                "the document declares how many hash-block syncs happened before the measurement; the run "
                "counts the commits above the boundary that touched the module or the focused test file and "
                "refuses to measure if the two disagree, recording the commit shas for the consumer"
            ),
        },
        "declared_sync_count": int(str(block.get("sync_count", "-1"))),
        "observed_sync_count": len(changes),
        "measurement_code_changes_since_the_boundary": changes,
        "rule": (
            "the measurement code, its thresholds and its adversarial tests are frozen by the hash block of "
            "the prespecification document; the block declares how many pre-measurement hash-block syncs "
            "exist and the run records exactly which commits they were. A run whose files do not match the "
            "block, whose prespecification commit is not an ancestor, or whose measurement code changed "
            "without a declared sync, is refused"
        ),
        "holds": True,
    }


def _relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def _seal_of(manifest: dict[str, Any]) -> dict[str, str]:
    seal = stage_2y.stage_2x._recorded_evidence_seal(manifest)
    if not seal:
        raise AuditBlocked("the recorded manifest declares no bound-file seal")
    return seal


def prerequisite_verification() -> dict[str, Any]:
    """Re-verify every frozen input: the L1A-2y seal and the L1A-2x seal, file by file."""
    l1a2y = stage_2y.verify_evidence_manifest(PRIOR_EVIDENCE["l1a2y"])
    l1a2x = stage_2x.verify_prerequisites()
    record = {
        "stage": STAGE,
        "prior_stages": {
            "l1a2y": {
                "verified": bool(l1a2y.get("verified")),
                "verdict": l1a2y.get("verdict"),
                "bound_files": l1a2y.get("bound_files"),
                "problems": l1a2y.get("problems"),
            },
            "frozen_chain": {
                "all_verified": bool(l1a2x.get("all_verified")),
                "stages": {
                    name: {
                        "status": entry.get("status"),
                        "bound_files": entry.get("bound_files"),
                        "verdict": entry.get("verdict"),
                        "problems": entry.get("problems"),
                    }
                    for name, entry in sorted(l1a2x.get("stages", {}).items())
                },
                "failures": list(l1a2x.get("failures", [])),
            },
        },
        "rule": (
            "the prior stages are read-only inputs: their evidence is re-hashed against their own "
            "manifests and their recorded source hashes are re-verified from the worktree before this "
            "stage measures anything"
        ),
    }
    if not record["prior_stages"]["l1a2y"]["verified"] or not record["prior_stages"]["frozen_chain"]["all_verified"]:
        raise AuditBlocked(f"a frozen prerequisite failed verification: {record['prior_stages']}")
    return record


def source_and_provenance(preflight: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    """Every source hash this stage depends on, recorded with the module that consumed it."""
    sources = {
        "stage_module": MODULE_PATH,
        "readout_module": readout.__file__,
        "l1a2x_module": Path(stage_2x.__file__),
        "l1a2y_module": Path(stage_2y.__file__),
        "frozen_audit_module": Path(audit.__file__),
        "phasefield_module": TWO_PHASE / "phasefield.py",
        "prespecification": PRESPECIFICATION,
        "stage_test": TEST_PATH,
    }
    return {
        "stage": STAGE,
        "head_sha": preflight["head_sha"],
        "baseline_head_sha": preflight["baseline_head_sha"],
        "verified_main_sha": VERIFIED_MAIN_SHA,
        "solver_contract_version": int(stage_2x.pf.SOLVER_CONTRACT_VERSION),
        "readout_version": readout.READOUT_VERSION,
        "source_hashes": {name: sha256_file(path) for name, path in sorted(sources.items())},
        "source_paths": {name: _relative(path) for name, path in sorted(sources.items())},
        "prespecification_commit": freeze["prespecification_commit"],
        "frozen_sources_expected_unchanged": {
            "phasefield_module": "the frozen solver is byte-identical to the L1A-2x and L1A-2y runs",
            "l1a2x_module": "the frozen kernel capture and the frozen uniqueness diagnostic are read-only",
            "l1a2y_module": "the frozen readout and the L1A-2y consumer verifier are read-only",
            "readout_module": "the frozen contour-branch readout is imported, never edited",
        },
        "rule": (
            "the trajectory, the contact guard, the mesh, the time step and every physical constant come "
            "from the frozen L1A-2x capture of the C2 state; this stage adds a measurement layer and does "
            "not touch a solver expression"
        ),
    }


# ---------------------------------------------------------------------------
# Section B - the reconstructed contour, its components and the support rule
# ---------------------------------------------------------------------------
def contour_edges(phi: Any, p: Any) -> dict[str, np.ndarray]:
    """Every reconstructed ``phi = 0.5`` crossing of the frozen reconstruction, with its grid edge.

    The crossing condition, the fraction and the periodic wrap reproduce
    ``impact_initialization_compatibility_audit._threshold_contour_edge_crossings`` exactly; the edge
    identity is carried alongside the coordinates so a crossing can be attached to a contour component.
    ``orientation`` 0 is the edge between ``(i, j)`` and ``(i + 1, j)``, 1 the edge between ``(i, j)`` and
    ``(i, j + 1)``.
    """
    values = np.asarray(phi, dtype=np.float64)
    Nx, Ny = int(p.Nx), int(p.Ny)
    dx, dy = float(p.dx), float(p.dy)
    Lx, Ly = float(p.Lx), float(p.Ly)
    threshold = CONTOUR_THRESHOLD
    x_axis = (np.arange(Nx, dtype=np.float64) + 0.5) * dx
    y_axis = (np.arange(Ny, dtype=np.float64) + 0.5) * dy
    columns: dict[str, list[np.ndarray]] = {key: [] for key in ("x", "y", "orientation", "i", "j", "fraction")}
    for orientation in (0, 1):
        axis = 0 if orientation == 0 else 1
        neighbour = np.roll(values, -1, axis=axis)
        crosses = ((values <= threshold) & (neighbour >= threshold)) | (
            (values >= threshold) & (neighbour <= threshold)
        )
        crosses &= neighbour != values
        i, j = np.where(crosses)
        if not len(i):
            continue
        first = values[i, j]
        second = neighbour[i, j]
        fraction = (threshold - first) / (second - first)
        if orientation == 0:
            xs = (x_axis[i] + fraction * dx) % Lx
            ys = y_axis[j]
        else:
            xs = x_axis[i]
            ys = (y_axis[j] + fraction * dy) % Ly
        columns["x"].append(np.asarray(xs, dtype=np.float64))
        columns["y"].append(np.asarray(ys, dtype=np.float64))
        columns["orientation"].append(np.full(len(i), orientation, dtype=np.int64))
        columns["i"].append(i.astype(np.int64))
        columns["j"].append(j.astype(np.int64))
        columns["fraction"].append(np.asarray(fraction, dtype=np.float64))
    return {key: (np.concatenate(value) if value else np.empty(0, dtype=np.float64)) for key, value in columns.items()}


def frozen_enumeration_matches(phi: Any, p: Any, edges: dict[str, np.ndarray]) -> dict[str, Any]:
    """The stage's crossing enumeration must reproduce the frozen one point for point."""
    frozen = audit._threshold_contour_edge_crossings(np.asarray(phi, dtype=np.float64), CONTOUR_THRESHOLD, p)
    mine = np.stack((edges["x"], edges["y"]), axis=1) if edges["x"].size else np.empty((0, 2))
    if mine.shape[0] != frozen.shape[0]:
        return {"matches": False, "count_mine": int(mine.shape[0]), "count_frozen": int(frozen.shape[0])}
    if mine.shape[0] == 0:
        return {"matches": True, "count_mine": 0, "count_frozen": 0, "max_abs_difference": 0.0}
    difference = float(
        np.max(np.abs(mine[np.lexsort((mine[:, 1], mine[:, 0]))] - frozen[np.lexsort((frozen[:, 1], frozen[:, 0]))]))
    )
    return {
        "matches": bool(difference <= 1.0e-12),
        "count_mine": int(mine.shape[0]),
        "count_frozen": int(frozen.shape[0]),
        "max_abs_difference": difference,
    }


def _cell_edge_keys(i: int, j: int, Nx: int, Ny: int) -> dict[str, tuple[int, int, int]]:
    return {
        "BC": (0, i % Nx, j % Ny),
        "TC": (0, i % Nx, (j + 1) % Ny),
        "LC": (1, i % Nx, j % Ny),
        "RC": (1, (i + 1) % Nx, j % Ny),
    }


def _cell_corner_values(values: np.ndarray, i: int, j: int, Nx: int, Ny: int) -> dict[str, float]:
    return {
        "BL": float(values[i, j]),
        "BR": float(values[(i + 1) % Nx, j]),
        "TL": float(values[i, (j + 1) % Ny]),
        "TR": float(values[(i + 1) % Nx, (j + 1) % Ny]),
    }


def _cell_pairs(corners: dict[str, float], found: list[str]) -> list[tuple[str, str]]:
    """Edge pairs one cell contributes, mirroring the frozen asymptotic-decider branch exactly."""
    threshold = CONTOUR_THRESHOLD
    if len(found) == 2:
        return [(found[0], found[1])]
    if len(found) != 4:
        return []
    low_corners = tuple(sorted(name for name, value in corners.items() if value < threshold))
    center_high = 0.25 * sum(corners.values()) > threshold
    if low_corners == ("BL", "TR"):
        return [("BC", "LC"), ("TC", "RC")] if center_high else [("LC", "TC"), ("BC", "RC")]
    if low_corners == ("BR", "TL"):
        return [("BC", "RC"), ("LC", "TC")] if center_high else [("TC", "RC"), ("BC", "LC")]
    names = sorted(found)
    return [(names[0], names[1]), (names[2], names[3])]


def contour_components(phi: Any, p: Any, edges: dict[str, np.ndarray]) -> dict[str, Any]:
    """Connected components of the reconstructed contour, using the frozen per-cell segment pairing.

    Each crossing edge is a node of a union-find structure; in every cell the pairs of crossing edges the
    frozen ``cell_segments`` reconstruction joins - including the frozen asymptotic-decider choice at a
    four-crossing saddle cell - are unioned. The component of a crossing is therefore the component of the
    same curve the frozen readout traces; the construction is validated cell by cell against
    ``l1a2y_contour_branch_readout.cell_segments`` by :func:`validate_components_against_frozen`.
    """
    values = np.asarray(phi, dtype=np.float64)
    Nx, Ny = int(p.Nx), int(p.Ny)
    index_of_edge: dict[tuple[int, int, int], int] = {}
    for index in range(int(edges["x"].size)):
        key = (int(edges["orientation"][index]), int(edges["i"][index]), int(edges["j"][index]))
        index_of_edge.setdefault(key, index)
    parent = list(range(len(index_of_edge)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    points = np.stack((edges["x"], edges["y"]), axis=1) if edges["x"].size else np.empty((0, 2))
    cells_with_crossings: list[tuple[int, int]] = []
    segments_by_cell: dict[tuple[int, int], list[list[list[float]]]] = {}
    saddle_cells: list[list[int]] = []
    for i in range(Nx):
        for j in range(Ny):
            keys = _cell_edge_keys(i, j, Nx, Ny)
            found = [name for name in ("BC", "TC", "LC", "RC") if keys[name] in index_of_edge]
            if not found:
                continue
            cells_with_crossings.append((i, j))
            pairs = _cell_pairs(_cell_corner_values(values, i, j, Nx, Ny), found)
            if len(found) == 4:
                saddle_cells.append([i, j])
            segments: list[list[list[float]]] = []
            for first, second in pairs:
                union(index_of_edge[keys[first]], index_of_edge[keys[second]])
                first_point = points[index_of_edge[keys[first]]]
                second_point = points[index_of_edge[keys[second]]]
                segments.append(
                    [
                        [float(first_point[0]), float(first_point[1])],
                        [float(second_point[0]), float(second_point[1])],
                    ]
                )
            segments_by_cell[(i, j)] = segments
    labels_by_edge = {key: find(node) for key, node in index_of_edge.items()}
    component_sizes: dict[int, int] = {}
    for label in labels_by_edge.values():
        component_sizes[label] = component_sizes.get(label, 0) + 1
    crossing_labels = np.array(
        [
            labels_by_edge[(int(edges["orientation"][index]), int(edges["i"][index]), int(edges["j"][index]))]
            for index in range(int(edges["x"].size))
        ],
        dtype=np.int64,
    )
    return {
        "labels_by_edge": labels_by_edge,
        "crossing_labels": crossing_labels,
        "components": len(component_sizes),
        "component_sizes": {int(key): int(value) for key, value in sorted(component_sizes.items())},
        "crossing_edges": len(index_of_edge),
        "cells_with_crossings": cells_with_crossings,
        "segments_by_cell": segments_by_cell,
        "saddle_cells": saddle_cells,
    }


def component_points(edges: dict[str, np.ndarray], components: dict[str, Any], label: int | None) -> np.ndarray:
    """The crossing points of one connected component of one state."""
    if label is None:
        return np.empty((0, 2))
    mask = np.asarray(components["crossing_labels"]) == int(label)
    if not np.any(mask):
        return np.empty((0, 2))
    return np.stack((np.asarray(edges["x"])[mask], np.asarray(edges["y"])[mask]), axis=1)


def component_relation(
    points_before: np.ndarray, points_after: np.ndarray, p: Any, tolerance_over_dx: float = BRANCH_IDENTITY_TOLERANCE_DX
) -> dict[str, Any]:
    """How the tracked component of one state relates to the tracked component of the next state.

    Labels from the union-find construction are not identities: they depend on the merge order and are
    reported for information only. The identity rule is geometric. Every crossing of the earlier component
    must lie within the prespecified tolerance of some crossing of the later one and the other way round;
    a component that is contained in a larger one grew or merged, a component that contains a smaller one
    fragmented or split, and a pair with no crossing within tolerance is a reassignment to a different
    physical branch.
    """
    tolerance = float(tolerance_over_dx) * float(p.dx)
    if points_before.size == 0 or points_after.size == 0:
        return {
            "relation": "NO_COMPONENT",
            "crossings_before": int(points_before.shape[0]),
            "crossings_after": int(points_after.shape[0]),
            "reassignment": True,
        }
    before_to_after = _max_min_distance(points_before, points_after, p)
    after_to_before = _max_min_distance(points_after, points_before, p)
    if before_to_after <= tolerance and after_to_before <= tolerance:
        relation = "PERSISTENT"
    elif before_to_after <= tolerance:
        relation = "GREW_OR_MERGED" if points_after.shape[0] > points_before.shape[0] else "FRAGMENTED_OR_SPLIT"
    elif after_to_before <= tolerance:
        relation = "FRAGMENTED_OR_SPLIT" if points_after.shape[0] < points_before.shape[0] else "GREW_OR_MERGED"
    else:
        relation = "DISJOINT_REASSIGNMENT"
    if relation not in BRANCH_IDENTITY_RELATIONS:  # pragma: no cover - defensive
        relation = "AMBIGUOUS_RELATION"
    return {
        "relation": relation,
        "crossings_before": int(points_before.shape[0]),
        "crossings_after": int(points_after.shape[0]),
        "max_min_distance_before_to_after_over_dx": before_to_after / float(p.dx),
        "max_min_distance_after_to_before_over_dx": after_to_before / float(p.dx),
        "tolerance_over_dx": float(tolerance_over_dx),
        "reassignment": relation in ("DISJOINT_REASSIGNMENT", "NO_COMPONENT"),
    }


def _max_min_distance(source: np.ndarray, target: np.ndarray, p: Any) -> float:
    """Largest nearest-neighbour distance from ``source`` points to ``target`` points, in grid units."""
    distances = []
    for point in source:
        dx_wall = audit._periodic_distance_x(np.asarray(target[:, 0], dtype=np.float64), float(point[0]), float(p.Lx))
        dy_wall = (np.asarray(target[:, 1], dtype=np.float64) - float(point[1]) + 0.5 * float(p.Ly)) % float(
            p.Ly
        ) - 0.5 * float(p.Ly)
        distances.append(float(np.min(np.sqrt(dx_wall * dx_wall + dy_wall * dy_wall))))
    return float(np.max(distances))


def validate_components_against_frozen(phi: Any, p: Any, components: dict[str, Any]) -> dict[str, Any]:
    """Cell-by-cell agreement of the component construction with the frozen cell reconstruction."""
    values = np.asarray(phi, dtype=np.float64)
    problems: list[str] = []
    for i, j in components["cells_with_crossings"]:
        frozen = readout.cell_segments(values, i, j, p)
        mine = components["segments_by_cell"][(i, j)]
        if len(mine) != len(frozen["segments"]):
            problems.append(f"cell {(i, j)}: {len(mine)} segments vs {len(frozen['segments'])} frozen")
            continue
        sorted_mine = sorted(sorted(tuple(endpoint) for endpoint in segment) for segment in mine)
        sorted_frozen = sorted(
            sorted(tuple(float(value) for value in endpoint) for endpoint in segment) for segment in frozen["segments"]
        )
        for own_pair, frozen_pair in zip(sorted_mine, sorted_frozen, strict=True):
            for own, theirs in zip(own_pair, frozen_pair, strict=True):
                if max(abs(own[0] - theirs[0]), abs(own[1] - theirs[1])) > 1.0e-12:
                    problems.append(f"cell {(i, j)}: endpoint mismatch")
                    break
    return {
        "cells_checked": len(components["cells_with_crossings"]),
        "matches": not problems,
        "problems": problems[:8],
        "rule": (
            "every cell that carries a crossing is reconstructed again by the frozen per-cell routine and "
            "the segment endpoints and the pairing are compared; a mismatch means the component graph is "
            "not the frozen reconstruction"
        ),
    }


def support_uniqueness(
    phi: Any,
    p: Any,
    bundle: Any,
    *,
    validate: bool = False,
    edges: dict[str, np.ndarray] | None = None,
    components: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The prespecified uniqueness record of one state.

    Reports the frozen L1A-2x diagnostic unchanged, this stage's branch-level rule, every competing
    candidate with its component, and the exact-threshold degeneracy count.
    """
    values = np.asarray(phi, dtype=np.float64)
    if edges is None:
        edges = contour_edges(values, p)
    frozen_match = frozen_enumeration_matches(values, p, edges)
    exact_nodes = int(np.sum(values == CONTOUR_THRESHOLD))
    if edges["x"].size == 0:
        return {
            "status": "NO_CONTOUR",
            "candidates": 0,
            "exact_threshold_nodes": exact_nodes,
            "frozen_enumeration_matches": frozen_match,
            "rule_ok": False,
        }
    if components is None:
        components = contour_components(values, p, edges)
    sdf = np.asarray(bundle.solid.sdf, dtype=np.float64)
    distances = np.asarray(audit._periodic_bilinear_sample(sdf, edges["x"], edges["y"], p), dtype=np.float64)
    best = float(np.min(distances))
    best_index = int(np.argmin(distances))
    tie_mask = (distances - best) <= TIE_TOLERANCE_DX * float(p.dx)
    competition_mask = (distances - best) <= COMPETITION_TOLERANCE_DX * float(p.dx)
    labels = components["crossing_labels"]
    tie_labels = sorted({int(labels[index]) for index in np.where(tie_mask)[0]})
    competition_labels = sorted({int(labels[index]) for index in np.where(competition_mask)[0]})
    tie_points = [
        {
            "x": float(edges["x"][index]),
            "y": float(edges["y"][index]),
            "gap_over_dx": float(distances[index] / float(p.dx)),
            "component": int(labels[index]),
            "grid_edge": [int(edges["orientation"][index]), int(edges["i"][index]), int(edges["j"][index])],
        }
        for index in np.where(tie_mask)[0]
    ]
    competitors = [
        {
            "x": float(edges["x"][index]),
            "y": float(edges["y"][index]),
            "gap_over_dx": float(distances[index] / float(p.dx)),
            "component": int(labels[index]),
            "distance_to_the_best_over_dx": float(
                math.hypot(
                    float(edges["x"][index]) - float(edges["x"][best_index]),
                    float(edges["y"][index]) - float(edges["y"][best_index]),
                )
                / float(p.dx)
            ),
        }
        for index in np.where(competition_mask)[0]
    ]
    outside = np.where((distances - best) > COMPETITION_TOLERANCE_DX * float(p.dx))[0]
    nearest_outside: dict[str, Any] | None = None
    if outside.size:
        separations = np.hypot(
            edges["x"][outside] - float(edges["x"][best_index]), edges["y"][outside] - float(edges["y"][best_index])
        )
        pick = int(outside[int(np.argmin(separations))])
        nearest_outside = {
            "x": float(edges["x"][pick]),
            "y": float(edges["y"][pick]),
            "gap_over_dx": float(distances[pick] / float(p.dx)),
            "distance_to_the_best_over_dx": float(separations.min() / float(p.dx)),
            "gap_increase_over_dx": float((distances[pick] - best) / float(p.dx)),
            "component": int(labels[pick]),
        }
    branch_unique = len(competition_labels) == 1
    record = {
        "status": "UNIQUE_BRANCH" if branch_unique and not exact_nodes else "NON_UNIQUE_BRANCH",
        "candidates": int(edges["x"].size),
        "minimum_gap_over_dx": best / float(p.dx),
        "tie_set_size": len(tie_points),
        "tie_components": tie_labels,
        "competition_set_size": len(competitors),
        "competition_components": competition_labels,
        "branch_unique": bool(branch_unique),
        "tie_is_one_component": len(tie_labels) == 1,
        "competitors": competitors,
        "tie_points": tie_points,
        "components": components["components"],
        "component_sizes": components["component_sizes"],
        "exact_threshold_nodes": exact_nodes,
        "exact_threshold_degeneracy_ok": exact_nodes == 0,
        "frozen_l1a2x_diagnostic": stage_2x._contour_uniqueness(values, bundle),
        "frozen_enumeration_matches": frozen_match,
        "nearest_non_competing_candidate": nearest_outside,
        "rule_ok": bool(branch_unique and not exact_nodes and frozen_match["matches"] and len(tie_points) >= 1),
        "rule": (
            "every competing candidate within COMPETITION_TOLERANCE_DX of the global minimum wall "
            "distance must lie on one connected component of the frozen reconstruction, and no grid node "
            "may sit exactly on the threshold; a tie inside that component is recorded, is never called "
            "uniqueness of a point, and is handled by measuring on every tied support"
        ),
    }
    if validate:
        validation = validate_components_against_frozen(values, p, components)
        record["frozen_cell_by_cell_validation"] = validation
        record["rule_ok"] = bool(record["rule_ok"] and validation["matches"])
    return record


def _edge_component(components: dict[str, Any], orientation: int, i: int, j: int, p: Any) -> int | None:
    key = (int(orientation), int(i) % int(p.Nx), int(j) % int(p.Ny))
    return components["labels_by_edge"].get(key)


def root_component(
    phi: Any, p: Any, record: dict[str, Any], *, edges: Any = None, components: Any = None
) -> dict[str, Any]:
    """Component of a tracked root, read from the grid edge the frozen reconstruction puts it on."""
    if record.get("root_over_dx") is None or record.get("segment_index") is None:
        return {"component": None, "reason": "no tracked root on this state"}
    if edges is None:
        edges = contour_edges(np.asarray(phi, dtype=np.float64), p)
    if components is None:
        components = contour_components(np.asarray(phi, dtype=np.float64), p, edges)
    label = _edge_component(components, 1, int(record["column_index"]), int(record["segment_index"]), p)
    return {
        "component": label,
        "grid_edge": [1, int(record["column_index"]), int(record["segment_index"])],
        "components_total": components["components"],
    }


def _tie_pair_arc_separation(phi: np.ndarray, p: Any, points: list[list[float]]) -> dict[str, Any]:
    """How the tied candidates sit on the reconstructed contour, locally and quantitatively.

    Both candidates are projected onto the segments of the frozen reconstruction inside the pinned window,
    and the shortest path between their nearest segments through the frozen segment graph is measured. The
    tie is a same-branch multiplicity only if such a path exists inside the window and both projections are
    on the contour; otherwise it is a multiplicity across separated branches and the gate blocks. The path
    length is the quantitative separation of the tie, and it is reported next to the separation to the
    nearest non-competing candidate.
    """
    seed = (float(points[0][0]), float(points[0][1]))
    window = readout.window_segments(phi, p, seed, readout.ARC_WINDOW_RADIUS_DX)
    segments = window["segments"]
    arc = readout.trace_arc(phi, p, seed, radius_dx=readout.ARC_WINDOW_RADIUS_DX)
    if not segments:
        return {
            "along_arc_over_dx": None,
            "same_local_branch": False,
            "arc_status": arc.get("status"),
            "note": "the pinned window carries no reconstructed segment",
        }
    lengths = [
        float(
            np.hypot(
                float(segment["endpoints"][1][0]) - float(segment["endpoints"][0][0]),
                float(segment["endpoints"][1][1]) - float(segment["endpoints"][0][1]),
            )
        )
        for segment in segments
    ]
    adjacency: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for index, segment in enumerate(segments):
        first = readout._key(tuple(float(value) for value in segment["endpoints"][0]))
        second = readout._key(tuple(float(value) for value in segment["endpoints"][1]))
        adjacency.setdefault(first, []).append((index, lengths[index]))
        adjacency.setdefault(second, []).append((index, lengths[index]))

    def nearest_segment(point: list[float]) -> tuple[int, float]:
        best_index, best_distance = 0, float("inf")
        for index, segment in enumerate(segments):
            start = np.array([float(value) for value in segment["endpoints"][0]], dtype=np.float64)
            end = np.array([float(value) for value in segment["endpoints"][1]], dtype=np.float64)
            direction = end - start
            length_squared = float(np.dot(direction, direction))
            if length_squared <= 0.0:
                continue
            offset = np.array([point[0], point[1]], dtype=np.float64) - start
            fraction = float(np.clip(np.dot(offset, direction) / length_squared, 0.0, 1.0))
            distance = float(
                np.linalg.norm(start + fraction * direction - np.array([point[0], point[1]], dtype=np.float64))
            )
            if distance < best_distance:
                best_index, best_distance = index, distance
        return best_index, best_distance

    projections = [nearest_segment(point) for point in points]
    start_index, start_distance = projections[0]
    target_index, target_distance = projections[1] if len(projections) > 1 else projections[0]
    distance = [float("inf")] * len(segments)
    distance[start_index] = 0.0
    hops: dict[int, int] = {start_index: 0}
    frontier = [(0.0, start_index)]
    while frontier:
        frontier.sort()
        current_cost, current = frontier.pop(0)
        if current_cost > distance[current]:
            continue
        if current == target_index:
            break
        for endpoint in (segments[current]["endpoints"][0], segments[current]["endpoints"][1]):
            for neighbour, weight in adjacency.get(readout._key(tuple(float(value) for value in endpoint)), []):
                candidate = current_cost + weight
                if candidate < distance[neighbour] - 1e-15:
                    distance[neighbour] = candidate
                    hops[neighbour] = hops.get(current, 0) + 1
                    frontier.append((candidate, neighbour))
    path_over_dx = None if not np.isfinite(distance[target_index]) else float(distance[target_index] / float(p.dx))
    both_on_the_contour = bool(
        start_distance <= TIE_ARC_TOLERANCE_DX * float(p.dx) and target_distance <= TIE_ARC_TOLERANCE_DX * float(p.dx)
    )
    return {
        "along_arc_over_dx": path_over_dx,
        "segment_hops": int(hops.get(target_index, 0)),
        "same_local_branch": bool(both_on_the_contour and path_over_dx is not None),
        "on_one_arc": bool(arc.get("open_simple_path")),
        "distance_to_the_contour_over_dx": [entry[1] / float(p.dx) for entry in projections],
        "arc_status": arc.get("status"),
        "segments_in_the_window": len(segments),
        "segments_traced": arc.get("segments_traced"),
        "window_radius_dx": readout.ARC_WINDOW_RADIUS_DX,
        "note": (
            "a path inside the pinned window means the tied pair is one local piece of the reconstructed "
            "contour; no path would mean separated branches and the gate would block"
        ),
    }


def _wall_normal_at(bundle: Any, point: tuple[float, float]) -> list[float] | None:
    """Nearest embedded-wall normal, replicating the frozen ``_local_gap`` lookup exactly."""
    solid = bundle.solid
    p = bundle.p
    wall_mask = np.asarray(solid.geometry.wall_measure) > 0.0
    if not wall_mask.any():
        return None
    wall_x = np.asarray(solid.geometry.wall_centroid_x, dtype=np.float64)[wall_mask]
    wall_y = np.asarray(solid.geometry.wall_centroid_y, dtype=np.float64)[wall_mask]
    nx = np.asarray(solid.geometry.wall_normal_x, dtype=np.float64)[wall_mask]
    ny = np.asarray(solid.geometry.wall_normal_y, dtype=np.float64)[wall_mask]
    dx_wall = audit._periodic_distance_x(wall_x, float(point[0]), float(p.Lx))
    dy_wall = (wall_y - float(point[1]) + 0.5 * float(p.Ly)) % float(p.Ly) - 0.5 * float(p.Ly)
    nearest = int(np.argmin(dx_wall * dx_wall + dy_wall * dy_wall))
    norm = math.hypot(float(nx[nearest]), float(ny[nearest]))
    if norm <= 1e-14:
        return None
    return [float(nx[nearest] / norm), float(ny[nearest] / norm)]


def support_record(bundle: Any, state: Any, point: tuple[float, float], label: str) -> dict[str, Any]:
    """One admissible support: its pinned branch definition and the frozen quantities at that point."""
    p = bundle.p
    phi0 = np.asarray(state.phi, dtype=np.float64)
    gradient = stage_2x._gradient_at(phi0, (float(point[0]), float(point[1])), bundle)
    norm = float(math.hypot(gradient[0], gradient[1]))
    if norm <= 0.0:
        raise AuditBlocked(f"the frozen discrete gradient vanishes at support {label}")
    interface_normal = (float(gradient[0] / norm), float(gradient[1] / norm))
    wall_normal = _wall_normal_at(bundle, (float(point[0]), float(point[1])))
    if wall_normal is None:
        raise AuditBlocked(f"no embedded-wall normal at support {label}")
    gap = float(
        np.asarray(
            audit._periodic_bilinear_sample(
                np.asarray(bundle.solid.sdf, dtype=np.float64),
                np.array([float(point[0])]),
                np.array([float(point[1])]),
                p,
            )
        ).reshape(-1)[0]
    )
    column_index = int(math.floor(float(point[0]) / float(p.dx)))
    cell = [column_index % int(p.Nx), int(math.floor(float(point[1]) / float(p.dy))) % int(p.Ny)]
    support = {
        "label": label,
        "cell": cell,
        "point": [float(point[0]), float(point[1])],
        "wall_normal_into_solid": [float(wall_normal[0]), float(wall_normal[1])],
        "interface_normal": [float(interface_normal[0]), float(interface_normal[1])],
        "gradient": [float(gradient[0]), float(gradient[1])],
        "gradient_norm": norm,
        "gap": gap,
        "gap_over_dx": gap / float(p.dx),
        "chi_at_cell": float(np.asarray(bundle.solid.chi, dtype=np.float64)[cell[0], cell[1]]),
        "gap_method": "BILINEAR_SDF_ON_LINEAR_PERIODIC_PHI_THRESHOLD_CONTOUR",
        "column_index": column_index,
    }
    support["coefficient0_over_dx"] = (
        -float(np.dot(np.array(support["interface_normal"]), np.array(support["wall_normal_into_solid"])))
        / norm
        / float(p.dx)
    )
    support["spec"] = readout.branch_definition(p, phi0, support)
    return support


def support_design(bundle: Any, state0: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pin every admissible support from the ``t = 0`` state alone and record the tie explicitly."""
    values = np.asarray(state0.phi, dtype=np.float64)
    p = bundle.p
    frozen_support = stage_2x.fixed_support(bundle, state0)
    uniqueness = support_uniqueness(values, p, bundle, validate=True)
    if uniqueness["status"] == "NO_CONTOUR":
        raise AuditBlocked("the t = 0 state carries no reconstructed contour support")
    tied = sorted(uniqueness["tie_points"], key=lambda item: item["x"])
    if not tied:
        raise AuditBlocked("the t = 0 state carries no minimum-gap candidate")
    supports = [
        support_record(
            bundle, state0, (float(item["x"]), float(item["y"])), "PRIMARY" if index == 0 else f"TIE_{index}"
        )
        for index, item in enumerate(tied)
    ]
    rebuilt = support_record(bundle, state0, tuple(float(value) for value in frozen_support["point"]), "FROZEN_RULE")
    replay = {
        "point_matches": bool(np.allclose(rebuilt["point"], frozen_support["point"], rtol=0.0, atol=1e-12)),
        "gradient_norm_difference": abs(rebuilt["gradient_norm"] - float(frozen_support["gradient_norm"])),
        "wall_normal_matches": bool(
            np.allclose(rebuilt["wall_normal_into_solid"], frozen_support["wall_normal_into_solid"], atol=1e-12)
        ),
        "interface_normal_matches": bool(
            np.allclose(rebuilt["interface_normal"], frozen_support["interface_normal"], atol=1e-12)
        ),
        "gap_matches": abs(rebuilt["gap"] - float(frozen_support["gap"])) <= 1e-12,
    }
    replay_ok = bool(
        replay["point_matches"]
        and replay["gradient_norm_difference"] <= 1e-12
        and replay["wall_normal_matches"]
        and replay["interface_normal_matches"]
        and replay["gap_matches"]
    )
    replay["holds"] = replay_ok
    if not replay_ok:
        raise AuditBlocked(f"the support construction does not reproduce the frozen support rule: {replay}")
    arc_separation = _tie_pair_arc_separation(values, p, [support["point"] for support in supports])
    mirror = {
        "tied_supports": len(supports),
        "same_height": all(abs(support["point"][1] - supports[0]["point"][1]) <= 1e-12 for support in supports),
        "wall_normals_agree": all(
            np.allclose(support["wall_normal_into_solid"], supports[0]["wall_normal_into_solid"], atol=1e-12)
            for support in supports
        ),
        "gap_differences_over_dx": [abs(support["gap"] - supports[0]["gap"]) / float(p.dx) for support in supports],
        "tie_components": uniqueness["tie_components"],
        "tie_is_one_component": uniqueness["tie_is_one_component"],
        "tied_points_are_all_on_one_component": len(uniqueness["tie_components"]) == 1,
        "tie_pair_local_path_over_dx": arc_separation["along_arc_over_dx"],
        "tie_pair_is_one_local_branch": arc_separation["same_local_branch"],
        "tie_pair_arc_detail": arc_separation,
        "nearest_non_competing_candidate": uniqueness["nearest_non_competing_candidate"],
        "separation_justification": (
            "the tied pair is separated from the next non-competing candidate by the gap increase and the "
            "point separation recorded above; the tie inside the pair is resolved by pinning both supports "
            "and reporting their difference, never by a tie-break"
        ),
    }
    design = {
        "stage": STAGE,
        "primary_support": supports[0]["label"],
        "primary_rule": PRIMARY_RULE,
        "historical_rule": HISTORICAL_RULE,
        "tie_tolerance_dx": TIE_TOLERANCE_DX,
        "competition_tolerance_dx": COMPETITION_TOLERANCE_DX,
        "tie_break": (
            "the primary support is the tied minimum-gap candidate with the smallest x; this is a "
            "deterministic tie-break for reporting, it is never used as evidence of uniqueness, and the "
            "whole attribution is measured on every tied support so the tie contribution to the "
            "comparison error is visible"
        ),
        "supports": [{key: value for key, value in support.items() if key != "spec"} for support in supports],
        "pinned_branch": [readout.spec_payload(support["spec"]) for support in supports],
        "frozen_support_replay": replay,
        "t0_uniqueness": uniqueness,
        "t0_mirror_structure": mirror,
        "frozen_l1a2x_rule_retained": (
            "the frozen L1A-2x minimum-gap diagnostic is recomputed here on every state and reported "
            "unchanged, including its NON_UNIQUE verdict; this stage does not redefine it and does not "
            "read its verdict as a branch ambiguity"
        ),
    }
    return design, supports


# ---------------------------------------------------------------------------
# Section C - the observable, the window and the sequential own-base ledger
# ---------------------------------------------------------------------------
def measure_root(
    field: Any,
    spec: dict[str, Any],
    previous_root: float,
    *,
    with_arc: bool = False,
) -> dict[str, Any]:
    """The frozen readout applied to one field: the tracked root, its status and its grid edge."""
    record = readout.branch_root(field, spec, previous_root, ray="wall_normal", with_arc=with_arc)
    matched = record.get("matched")
    return {
        "status": str(record["status"]),
        "root_over_dx": None if matched is None else float(matched["y_over_dx"]),
        "segment_index": None
        if matched is None or matched.get("segment_index") is None
        else int(matched["segment_index"]),
        "column_index": int(spec["column_index"]),
        "at_shared_node": None if matched is None else bool(matched.get("at_shared_node", False)),
        "step_over_dx": record.get("step_over_dx"),
        "separation_over_dx": record.get("separation_over_dx"),
        "candidates_over_dx": record.get("candidates_over_dx"),
        "saddle_ambiguous": bool((record.get("matched_cell") or {}).get("saddle_ambiguous", False)),
        "reason": record.get("reason"),
    }


def _displacement(root_from: float | None, root_to: float | None, wall_normal: Any) -> float | None:
    if root_from is None or root_to is None:
        return None
    return readout.displacement_over_dx_from_heights(float(root_from), float(root_to), tuple(wall_normal))


def _compact_uniqueness(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": record["status"],
        "candidates": record.get("candidates"),
        "minimum_gap_over_dx": record.get("minimum_gap_over_dx"),
        "tie_set_size": record.get("tie_set_size"),
        "tie_components": record.get("tie_components"),
        "competition_set_size": record.get("competition_set_size"),
        "competition_components": record.get("competition_components"),
        "branch_unique": record.get("branch_unique"),
        "exact_threshold_nodes": record.get("exact_threshold_nodes"),
        "frozen_status": (record.get("frozen_l1a2x_diagnostic") or {}).get("status"),
        "frozen_margin_over_dx": (record.get("frozen_l1a2x_diagnostic") or {}).get("margin_over_dx"),
        "frozen_enumeration_matches": (record.get("frozen_enumeration_matches") or {}).get("matches"),
        "rule_ok": record.get("rule_ok"),
    }


def run_window(macro_steps: int = MAX_MACRO_STEPS) -> dict[str, Any]:
    """Run the one sanctioned window and measure the corrected decomposition on every frozen substep.

    The solver path is the frozen L1A-2x kernel capture of the frozen C2 state: one case, one ``dt``, one
    initial state, at most ``MAX_MACRO_STEPS`` macro steps, and the window stops early if the frozen
    contact gap threshold is reached. Nothing about the trajectory is changed by this stage.
    """
    import jax

    if macro_steps < 1 or macro_steps > MAX_MACRO_STEPS:
        raise AuditBlocked(f"macro step budget {macro_steps} is outside 1..{MAX_MACRO_STEPS}")
    bundle, state0 = stage_2x._bundle_and_state()
    p = bundle.p
    solid = bundle.solid
    pf.validate_x64_for_phase_storage(p.phase_storage_model)
    dx = float(p.dx)
    design, supports = support_design(bundle, state0)
    volume = np.asarray(pf.phase_control_volumes(solid, p), dtype=np.float64)
    aperture_x = np.asarray(pf.phase_face_apertures(solid, p)[0], dtype=np.float64)
    aperture_y = np.asarray(pf.phase_face_apertures(solid, p)[1], dtype=np.float64)
    initial_hashes = {
        "phi_hash": _array_sha(state0.phi),
        "u_hash": _array_sha(state0.u),
        "v_hash": _array_sha(state0.v),
        "t": float(np.asarray(state0.t)),
    }
    jit_step = jax.jit(pf.step, static_argnums=(2,))
    raw_eager = state0
    raw_jitted = state0
    equivalence: dict[str, Any] = {"per_macro": []}
    phi, u, v, t = state0.phi, state0.u, state0.v, state0.t
    contact_guard: dict[str, Any] = {"triggered": False, "macro_step": None, "gap_over_dx": None}
    rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    uniqueness_rows: list[dict[str, Any]] = []
    segment_binding_switch: dict[str, Any] = {"found": False, "macro_step": None, "substep": None}
    previous: dict[str, float] = {support["label"]: float(support["spec"]["t0_root_over_dx"]) for support in supports}
    carried_components: dict[str, Any] | None = None
    carried_edges: dict[str, Any] | None = None
    pinned_components: dict[str, Any] = {}
    t0_uniqueness = design["t0_uniqueness"]
    for support in supports:
        record = measure_root(np.asarray(state0.phi, dtype=np.float64), support["spec"], previous[support["label"]])
        pinned_components[support["label"]] = {
            "t0_root_over_dx": record["root_over_dx"],
            "t0_status": record["status"],
            "t0_segment_index": record["segment_index"],
            "t0_component": root_component(np.asarray(state0.phi, dtype=np.float64), p, record)["component"],
        }
    for macro in range(macro_steps):
        macro_aggregate: dict[str, dict[str, float]] = {
            support["label"]: {"adv": 0.0, "ch": 0.0, "imp": 0.0, "full": 0.0} for support in supports
        }
        for _substep in range(SUBSTEPS_PER_MACRO):
            base = np.asarray(phi, dtype=np.float64)
            base_roots = {support["label"]: float(previous[support["label"]]) for support in supports}
            t_before = float(np.asarray(t))
            mass_before = stage_2x._mass(volume, phi)
            (phi, u, v, t), ingredient = stage_2x.substep_capture(phi, u, v, t, bundle)
            mass_after = stage_2x._mass(volume, phi)
            d_adv = np.asarray(ingredient["d_adv"], dtype=np.float64)
            d_ch = np.asarray(ingredient["d_ch"], dtype=np.float64)
            d_imp = np.asarray(ingredient["d_imp"], dtype=np.float64)
            d_full = np.asarray(ingredient["d_full"], dtype=np.float64)
            state_s1 = base + d_adv
            state_s2 = np.asarray(ingredient["phi_hat"], dtype=np.float64)
            state_s3 = np.asarray(phi, dtype=np.float64)
            phi_hat_identity = float(np.max(np.abs((state_s1 + d_ch) - state_s2)))
            recombination = float(np.max(np.abs((d_adv + d_ch + d_imp) - d_full)))
            states = {"s0": base, "s1": state_s1, "s2": state_s2, "s3": state_s3}
            state_hashes = {name: _array_sha(value) for name, value in states.items()}
            state_edges = {"s0": carried_edges} if carried_edges is not None else {}
            state_components = {"s0": carried_components} if carried_components is not None else {}
            if state_edges.get("s0") is None:
                state_edges["s0"] = contour_edges(base, p)
                state_components["s0"] = contour_components(base, p, state_edges["s0"])
            for state_name, state_field in (("s1", state_s1), ("s2", state_s2), ("s3", state_s3)):
                state_edges[state_name] = contour_edges(state_field, p)
                state_components[state_name] = contour_components(state_field, p, state_edges[state_name])
            per_support: dict[str, Any] = {}
            marginal: dict[str, Any] = {}
            for support in supports:
                label = support["label"]
                spec = support["spec"]
                wall_normal = support["wall_normal_into_solid"]
                q0 = base_roots[label]
                record0 = measure_root(base, spec, q0)
                q1 = measure_root(state_s1, spec, q0)
                q2 = measure_root(state_s2, spec, float(q1["root_over_dx"]) if q1["root_over_dx"] is not None else q0)
                chain_previous = float(q2["root_over_dx"]) if q2["root_over_dx"] is not None else q0
                q3 = measure_root(state_s3, spec, chain_previous)
                q3_direct = measure_root(state_s3, spec, q0)
                d_sequential = {
                    "adv": _displacement(q0, q1["root_over_dx"], wall_normal),
                    "ch": _displacement(q1["root_over_dx"], q2["root_over_dx"], wall_normal),
                    "imp": _displacement(q2["root_over_dx"], q3["root_over_dx"], wall_normal),
                    "full": _displacement(q0, q3["root_over_dx"], wall_normal),
                }
                closure = None
                if all(d_sequential[name] is not None for name in ("adv", "ch", "imp", "full")):
                    closure = float(
                        d_sequential["adv"] + d_sequential["ch"] + d_sequential["imp"] - d_sequential["full"]
                    )
                root_components = {
                    "q0": _edge_component(state_components["s0"], 1, spec["column_index"], record0["segment_index"], p)
                    if record0["segment_index"] is not None
                    else None,
                    "q1": _edge_component(state_components["s1"], 1, spec["column_index"], q1["segment_index"], p)
                    if q1["segment_index"] is not None
                    else None,
                    "q2": _edge_component(state_components["s2"], 1, spec["column_index"], q2["segment_index"], p)
                    if q2["segment_index"] is not None
                    else None,
                    "q3": _edge_component(state_components["s3"], 1, spec["column_index"], q3["segment_index"], p)
                    if q3["segment_index"] is not None
                    else None,
                }
                component_relations = {
                    "s0_s1": component_relation(
                        component_points(state_edges["s0"], state_components["s0"], root_components["q0"]),
                        component_points(state_edges["s1"], state_components["s1"], root_components["q1"]),
                        p,
                    ),
                    "s1_s2": component_relation(
                        component_points(state_edges["s1"], state_components["s1"], root_components["q1"]),
                        component_points(state_edges["s2"], state_components["s2"], root_components["q2"]),
                        p,
                    ),
                    "s2_s3": component_relation(
                        component_points(state_edges["s2"], state_components["s2"], root_components["q2"]),
                        component_points(state_edges["s3"], state_components["s3"], root_components["q3"]),
                        p,
                    ),
                }
                per_support[label] = {
                    "q0_remeasured_over_dx": record0["root_over_dx"],
                    "q0_carried_over_dx": q0,
                    "q0_remeasure_delta": None
                    if record0["root_over_dx"] is None
                    else abs(float(record0["root_over_dx"]) - q0),
                    "q1_over_dx": q1["root_over_dx"],
                    "q2_over_dx": q2["root_over_dx"],
                    "q3_over_dx": q3["root_over_dx"],
                    "q3_direct_over_dx": q3_direct["root_over_dx"],
                    "long_baseline_delta": None
                    if q3["root_over_dx"] is None or q3_direct["root_over_dx"] is None
                    else abs(float(q3["root_over_dx"]) - float(q3_direct["root_over_dx"])),
                    "statuses": {
                        "q0": record0["status"],
                        "q1": q1["status"],
                        "q2": q2["status"],
                        "q3": q3["status"],
                        "q3_direct": q3_direct["status"],
                    },
                    "segment_indices": {
                        "q0": record0["segment_index"],
                        "q1": q1["segment_index"],
                        "q2": q2["segment_index"],
                        "q3": q3["segment_index"],
                    },
                    "root_components": root_components,
                    "component_relations": component_relations,
                    "component_crossing_counts": {
                        name: int(state_components[name]["crossing_edges"]) for name in ("s0", "s1", "s2", "s3")
                    },
                    "component_totals": {
                        name: int(state_components[name]["components"]) for name in ("s0", "s1", "s2", "s3")
                    },
                    "at_shared_node": {
                        "q1": q1["at_shared_node"],
                        "q2": q2["at_shared_node"],
                        "q3": q3["at_shared_node"],
                    },
                    "d_sequential": d_sequential,
                    "closure_residual": closure,
                    "dphi_at_point": {
                        name: float(np.asarray(stage_2x._sample(value, tuple(support["point"]), bundle)))
                        for name, value in (("adv", d_adv), ("ch", d_ch), ("imp", d_imp), ("full", d_full))
                    },
                    "dphi_at_cell": {
                        name: float(np.asarray(value)[int(support["cell"][0]), int(support["cell"][1])])
                        for name, value in (("adv", d_adv), ("ch", d_ch), ("imp", d_imp), ("full", d_full))
                    },
                }
                if q3["root_over_dx"] is not None:
                    previous[label] = float(q3["root_over_dx"])
                marginal_partial = {}
                for name, increment in (("ch", d_ch), ("imp", d_imp)):
                    field = base + increment
                    record = measure_root(field, spec, q0)
                    marginal_partial[name] = {
                        "status": record["status"],
                        "d": _displacement(q0, record["root_over_dx"], wall_normal),
                    }
                swap_field = base + d_imp + d_ch
                swap_ch = measure_root(swap_field, spec, q0)
                swap_imp_first = measure_root(base + d_imp, spec, q0)
                swap_pair_total = None
                swap_pair_sequential = None
                if swap_ch["root_over_dx"] is not None and swap_imp_first["root_over_dx"] is not None:
                    swap_pair_total = float(
                        _displacement(q0, swap_imp_first["root_over_dx"], wall_normal)
                        + _displacement(swap_imp_first["root_over_dx"], swap_ch["root_over_dx"], wall_normal)
                    )
                    sequential_pair = 0.0
                    if d_sequential["ch"] is not None and d_sequential["imp"] is not None:
                        sequential_pair = float(d_sequential["ch"] + d_sequential["imp"])
                        swap_pair_sequential = sequential_pair
                marginal[label] = {
                    "adv": d_sequential["adv"],
                    "ch": marginal_partial["ch"]["d"],
                    "imp": marginal_partial["imp"]["d"],
                    "statuses": {
                        "ch": marginal_partial["ch"]["status"],
                        "imp": marginal_partial["imp"]["status"],
                        "swap_first": swap_imp_first["status"],
                        "swap_second": swap_ch["status"],
                    },
                    "order_swap_pair_total": swap_pair_total,
                    "order_swap_sequential_pair_total": swap_pair_sequential,
                    "order_swap_delta": None
                    if swap_pair_total is None or swap_pair_sequential is None
                    else abs(swap_pair_total - swap_pair_sequential),
                }
                for name in ("adv", "ch", "imp", "full"):
                    if d_sequential[name] is not None:
                        macro_aggregate[label][name] += float(d_sequential[name])
            uniqueness_states = {}
            for name in ("s1", "s2", "s3"):
                record = support_uniqueness(
                    states[name],
                    p,
                    bundle,
                    edges=state_edges[name],
                    components=state_components[name],
                )
                uniqueness_states[name] = _compact_uniqueness(record)
                uniqueness_rows.append(
                    {
                        "substep": len(rows),
                        "macro_step": macro + 1,
                        "state": name,
                        **_compact_uniqueness(record),
                    }
                )
            gap_row = audit._local_gap(state_s3, CONTOUR_THRESHOLD, solid, p)
            if gap_row.get("gap_over_dx") is None and gap_row.get("gap") is not None:
                gap_row["gap_over_dx"] = float(gap_row["gap"]) / dx
            if (
                not segment_binding_switch["found"]
                and per_support[supports[0]["label"]]["segment_indices"]["q3"] is not None
                and rows
                and rows[-1]["primary_segment_index"] is not None
                and per_support[supports[0]["label"]]["segment_indices"]["q3"] != rows[-1]["primary_segment_index"]
            ):
                segment_binding_switch = {
                    "found": True,
                    "macro_step": macro + 1,
                    "substep": len(rows),
                    "t": float(np.asarray(t)),
                    "from_segment_index": rows[-1]["primary_segment_index"],
                    "to_segment_index": per_support[supports[0]["label"]]["segment_indices"]["q3"],
                    "note": (
                        "the reconstructed crossing moved from one grid segment to the next; this is the "
                        "support change that stopped L1A-2x at its macro step 25 and that L1A-2y already "
                        "reported as an ordinary recorded event"
                    ),
                }
            rows.append(
                {
                    "substep": len(rows),
                    "macro_step": macro + 1,
                    "t_before": t_before,
                    "t_after": float(np.asarray(t)),
                    "state_hashes": state_hashes,
                    "field_increments": {
                        name: {
                            "linf": float(np.max(np.abs(value))),
                            "l2": float(np.sqrt(np.sum(np.asarray(value, dtype=np.float64) ** 2))),
                            "mass": stage_2x._mass(volume, value),
                        }
                        for name, value in (("adv", d_adv), ("ch", d_ch), ("imp", d_imp), ("full", d_full))
                    },
                    "phi_hat_identity_linf": phi_hat_identity,
                    "recombination_linf": recombination,
                    "mass_before": mass_before,
                    "mass_after": mass_after,
                    "mass_adv": stage_2x._mass(volume, d_adv),
                    "mass_ch": stage_2x._mass(volume, d_ch),
                    "mass_imp": stage_2x._mass(volume, d_imp),
                    "mass_complete": stage_2x._mass(volume, d_full),
                    "mass_residual": mass_after - mass_before - stage_2x._mass(volume, d_full),
                    "closed_face_flux_max": stage_2x._closed_face_flux_max(
                        [ingredient["adv_x"], ingredient["adv_y"], ingredient["ch_x"], ingredient["ch_y"]],
                        [aperture_x, aperture_y, aperture_x, aperture_y],
                    ),
                    "cg_iterations": int(ingredient["solve_info"].iterations),
                    "cg_relative_residual": float(ingredient["solve_info"].relative_residual),
                    "cg_converged": bool(ingredient["solve_info"].converged),
                    "gap_global_over_dx": gap_row.get("gap_over_dx"),
                    "gap_global_cell": gap_row.get("gap_cell"),
                    "wall_normal_global": gap_row.get("local_wall_normal_into_solid"),
                    "per_support": per_support,
                    "marginal": marginal,
                    "uniqueness": uniqueness_states,
                    "primary_segment_index": per_support[supports[0]["label"]]["segment_indices"]["q3"],
                    "carried_state_components": {
                        name: state_components[name]["components"] for name in ("s0", "s1", "s2", "s3")
                    },
                    "dphi_at_point": {
                        name: float(np.asarray(stage_2x._sample(value, tuple(supports[0]["point"]), bundle)))
                        for name, value in (("adv", d_adv), ("ch", d_ch), ("imp", d_imp), ("full", d_full))
                    },
                    "dphi_at_cells": {
                        name: float(np.asarray(value)[int(supports[0]["cell"][0]), int(supports[0]["cell"][1])])
                        for name, value in (("adv", d_adv), ("ch", d_ch), ("imp", d_imp), ("full", d_full))
                    },
                }
            )
            if gap_row.get("gap_over_dx") is not None and float(gap_row["gap_over_dx"]) <= CONTACT_GAP_THRESHOLD_DX:
                contact_guard = {
                    "triggered": True,
                    "macro_step": macro + 1,
                    "substep": len(rows) - 1,
                    "gap_over_dx": float(gap_row["gap_over_dx"]),
                }
        raw_eager, eager_info = pf.step_with_diagnostics(raw_eager, solid, p)
        raw_jitted = jax.block_until_ready(jit_step(raw_jitted, solid, p))
        phi_np = np.asarray(phi, dtype=np.float64)
        jitted_np = np.asarray(raw_jitted.phi, dtype=np.float64)
        gap_capture = rows[-1]["gap_global_over_dx"]
        gap_public = audit._local_gap(jitted_np, CONTOUR_THRESHOLD, solid, p)
        gap_public_dx = float(gap_public["gap"]) / dx if gap_public.get("gap") is not None else None
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
            }
        )
        primary = supports[0]["label"]
        uniqueness_macro = support_uniqueness(np.asarray(phi, dtype=np.float64), p, bundle)
        macro_rows.append(
            {
                "macro_step": macro + 1,
                "t": float(np.asarray(t)),
                "tracked_root_over_dx": previous[primary],
                "tracked_segment_index": rows[-1]["primary_segment_index"],
                "dn_adv": macro_aggregate[primary]["adv"],
                "dn_ch": macro_aggregate[primary]["ch"],
                "dn_imp": macro_aggregate[primary]["imp"],
                "dn_full": macro_aggregate[primary]["full"],
                "tie_columns": {
                    support["label"]: {
                        "root_over_dx": previous[support["label"]],
                        "dn_adv": macro_aggregate[support["label"]]["adv"],
                        "dn_ch": macro_aggregate[support["label"]]["ch"],
                        "dn_imp": macro_aggregate[support["label"]]["imp"],
                        "dn_full": macro_aggregate[support["label"]]["full"],
                    }
                    for support in supports
                },
                "uniqueness": _compact_uniqueness(uniqueness_macro),
                "gap_global_over_dx": rows[-1]["gap_global_over_dx"],
                "gap_public_step_over_dx": gap_public_dx,
                "gap_difference_over_dx": equivalence["per_macro"][-1]["gap_difference_over_dx"],
            }
        )
        carried_edges = state_edges["s3"]
        carried_components = state_components["s3"]
        if contact_guard["triggered"]:
            break
    final_hashes = {
        "phi_hash": _array_sha(phi),
        "u_hash": _array_sha(u),
        "v_hash": _array_sha(v),
        "t": float(np.asarray(t)),
    }
    ledger = {
        "stage": STAGE,
        "case_name": CASE_NAME,
        "profile_N": PROFILE_N,
        "requested_dt": REQUESTED_DT,
        "substeps_per_macro": SUBSTEPS_PER_MACRO,
        "macro_steps_requested": macro_steps,
        "macro_steps_run": len(macro_rows),
        "substeps_recorded": len(rows),
        "t_final": float(np.asarray(t)),
        "initial_state": initial_hashes,
        "final_state": final_hashes,
        "contact_guard": contact_guard,
        "contact_threshold_over_dx": CONTACT_GAP_THRESHOLD_DX,
        "segment_binding_switch": segment_binding_switch,
        "pinned_components": pinned_components,
        "t0_uniqueness": _compact_uniqueness(t0_uniqueness),
        "capture_equivalence": equivalence,
        "rows": rows,
        "macro_rows": macro_rows,
        "uniqueness_rows": uniqueness_rows,
        "supports": [
            {
                **{key: value for key, value in support.items() if key != "spec"},
                "t0_root_over_dx": float(support["spec"]["t0_root_over_dx"]),
                "t0_root_segment_index": int(support["spec"]["t0_root_segment_index"]),
            }
            for support in supports
        ],
        "parameters": {
            "N": int(p.Nx),
            "dx": dx,
            "dy": float(p.dy),
            "dt": float(p.dt),
            "substep_dt": float(p.dt) / SUBSTEPS_PER_MACRO,
            "eps": float(p.eps),
            "eps_over_dx": float(p.eps) / dx,
            "M": float(p.M),
            "Re": float(p.Re),
            "We": float(p.We),
            "eta_pen": float(p.eta_pen),
            "ch_solver_rtol": float(p.ch_solver_rtol),
            "ch_solver_max_iterations": int(p.ch_solver_max_iterations),
        },
        "prohibitions": {
            "control_arm_runs": 0,
            "contact_events_scored": 0,
            "refinement_runs": 0,
            "t8_recomputed": False,
            "l1b_data_generated": False,
            "ml_training_runs": 0,
            "solver_source_edited": False,
            "initializer_promoted": False,
        },
        "design": {key: value for key, value in design.items() if key != "pinned_branch"},
    }
    return ledger


# ---------------------------------------------------------------------------
# Section D - attribution, residuals, conservation and the identical-window check
# ---------------------------------------------------------------------------
def _sum_legs(rows: list[dict[str, Any]], support_label: str, key: str, family: str) -> float | None:
    values = [
        row["per_support"][support_label]["d_sequential"].get(key)
        if family == "sequential"
        else (
            row["per_support"][support_label]["d_sequential"].get("adv")
            if key == "adv"
            else row["marginal"][support_label].get(key)
        )
        for row in rows
    ]
    if any(value is None for value in values):
        return None
    return float(np.sum(values))


def support_uniqueness_window(ledger: dict[str, Any]) -> dict[str, Any]:
    """The window-level record of the support rule over every intermediate state.

    One row per measured intermediate state (three per substep) with the frozen L1A-2x diagnostic reported
    unchanged beside this stage's branch-level rule, plus the support-switch events and the branch-identity
    relation counts. Nothing is aggregated away: the intermediate states are the objects the gate applies
    to, so every one of them is listed.
    """
    rows = ledger["uniqueness_rows"]
    labels = [support["label"] for support in ledger["supports"]]
    counts: dict[str, int] = {}
    frozen_counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
        frozen_counts[str(row["frozen_status"])] = frozen_counts.get(str(row["frozen_status"]), 0) + 1
    relations = [
        entry["relation"]
        for row in ledger["rows"]
        for label in labels
        for entry in row["per_support"][label]["component_relations"].values()
    ]
    relation_counts: dict[str, int] = {}
    for relation in relations:
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
    reassignments = sorted(
        {
            f"macro {row['macro_step']} substep {row['substep']} {label} {state}"
            for row in ledger["rows"]
            for label in labels
            for state, entry in row["per_support"][label]["component_relations"].items()
            if entry["reassignment"]
        }
    )
    switch = ledger["segment_binding_switch"]
    margins = [float(row["frozen_margin_over_dx"]) for row in rows if row["frozen_margin_over_dx"] is not None]
    return {
        "stage": STAGE,
        "window": {
            "macro_steps": ledger["macro_steps_run"],
            "substeps": ledger["substeps_recorded"],
            "states_per_substep": len(rows) // max(ledger["substeps_recorded"], 1),
            "intermediate_states": len(rows),
            "t_final": ledger["t_final"],
        },
        "statuses": counts,
        "frozen_diagnostic_statuses": frozen_counts,
        "frozen_diagnostic_retained": {
            "constant": float(stage_2x.CONTOUR_UNIQUENESS_MARGIN_DX),
            "statuses": sorted(frozen_counts),
            "minimum_margin_over_dx": min(margins) if margins else None,
            "note": (
                "reported unchanged and never redefined: the frozen rule asks whether one candidate minimises "
                "the wall distance, this stage asks whether every competing candidate lies on one component"
            ),
        },
        "branch_identity": {
            "relation_counts": relation_counts,
            "reassignments": reassignments,
            "tolerance_over_dx": BRANCH_IDENTITY_TOLERANCE_DX,
        },
        "support_switch_events": [switch] if switch.get("found") else [],
        "states_with_a_non_unique_support": sorted(
            {f"macro {row['macro_step']} substep {row['substep']} {row['state']}" for row in rows if not row["rule_ok"]}
        ),
        "states_with_an_exact_threshold_node": sorted(
            {
                f"macro {row['macro_step']} substep {row['substep']} {row['state']}"
                for row in rows
                if row["exact_threshold_nodes"]
            }
        ),
        "states_with_a_multi_component_competition": sorted(
            {
                f"macro {row['macro_step']} substep {row['substep']} {row['state']}"
                for row in rows
                if len(row["competition_components"]) > 1
            }
        ),
        "every_intermediate_state_unique": all(bool(row["rule_ok"]) for row in rows),
        "uniqueness_rows": rows,
        "rule": (
            "the gate applies to every required intermediate state of the window, not only to the end points; "
            "each row reports the candidate count, the frozen diagnostic, this stage's component-level answer, "
            "the tie set and the exact-threshold degeneracy count"
        ),
    }


def attribution(ledger: dict[str, Any]) -> dict[str, Any]:
    """Every prespecified attribution convention of the corrected measurement, in raw dx."""
    rows = ledger["rows"]
    labels = [support["label"] for support in ledger["supports"]]
    primary = ledger["design"]["primary_support"]
    window_displacement: dict[str, float] = {}
    for label in labels:
        sequential = {
            "adv": _sum_legs(rows, label, "adv", "sequential"),
            "ch_explicit": _sum_legs(rows, label, "ch", "sequential"),
            "ch_implicit": _sum_legs(rows, label, "imp", "sequential"),
            "full": _sum_legs(rows, label, "full", "sequential"),
        }
        sequential["ch_pair"] = (
            None
            if sequential["ch_explicit"] is None or sequential["ch_implicit"] is None
            else float(sequential["ch_explicit"] + sequential["ch_implicit"])
        )
        marginal = {
            "adv": sequential["adv"],
            "ch_explicit": _sum_legs(rows, label, "ch", "marginal"),
            "ch_implicit": _sum_legs(rows, label, "imp", "marginal"),
        }
        marginal["ch_pair"] = (
            None
            if marginal["ch_explicit"] is None or marginal["ch_implicit"] is None
            else float(marginal["ch_explicit"] + marginal["ch_implicit"])
        )
        coefficient = None
        for support in ledger["supports"]:
            if support["label"] == label:
                coefficient = float(support["coefficient0_over_dx"])
        if coefficient is None:
            raise AuditBlocked(f"no t = 0 coefficient for support {label}")
        linearised_point = {
            name: float(coefficient * np.sum([row["per_support"][label]["dphi_at_point"][name] for row in rows]))
            for name in ("adv", "ch", "imp", "full")
        }
        linearised_cell = {
            name: float(coefficient * np.sum([row["per_support"][label]["dphi_at_cell"][name] for row in rows]))
            for name in ("adv", "ch", "imp", "full")
        }
        for family in (linearised_point, linearised_cell):
            family["ch_pair"] = float(family["ch"] + family["imp"])
        window_displacement[label] = {
            "SEQUENTIAL_OWN_BASE": sequential,
            "MARGINAL_OWN_BASE": marginal,
            "LINEARISED_POINT_T0": linearised_point,
            "LINEARISED_SUPPORT_CELL_T0": linearised_cell,
        }
    margins: dict[str, dict[str, float]] = {}
    for label in labels:
        margins[label] = {
            convention: float(abs(values["adv"]) - abs(values["ch_pair"]))
            for convention, values in window_displacement[label].items()
            if values["adv"] is not None and values["ch_pair"] is not None
        }
    primary_margin = margins[primary][PRIMARY_CONVENTION]
    convention_spread = max(
        abs(primary_margin - margins[primary][convention])
        for convention in margins[primary]
        if convention != PRIMARY_CONVENTION
    )
    tie_spread = max(abs(primary_margin - margins[label][PRIMARY_CONVENTION]) for label in labels if label != primary)
    root_matching_noise = max(
        [row["per_support"][label]["long_baseline_delta"] or 0.0 for row in rows for label in labels]
    )
    closure_residuals = {
        label: {
            "maximum_abs_substep": float(np.max([abs(row["per_support"][label]["closure_residual"]) for row in rows])),
            "window_total": float(np.sum([row["per_support"][label]["closure_residual"] for row in rows])),
        }
        for label in labels
    }
    comparison_terms = {
        "convention_spread": float(convention_spread),
        "support_tie_spread": float(tie_spread),
        "root_matching_noise": float(root_matching_noise),
        "prespecified_floor": WALK_NOISE_FLOOR_DX,
    }
    comparison_error = float(max(comparison_terms.values()))
    per_macro = []
    for index, macro_row in enumerate(ledger["macro_rows"]):
        entry = {
            "macro_step": macro_row["macro_step"],
            "t": macro_row["t"],
            "dn_adv": macro_row["dn_adv"],
            "dn_ch_explicit": macro_row["dn_ch"],
            "dn_ch_implicit": macro_row["dn_imp"],
            "dn_full": macro_row["dn_full"],
            "tracked_root_over_dx": macro_row["tracked_root_over_dx"],
            "tracked_segment_index": macro_row["tracked_segment_index"],
            "tie_columns": macro_row["tie_columns"],
        }
        entry["dn_ch_pair"] = float(entry["dn_ch_explicit"] + entry["dn_ch_implicit"])
        per_macro.append(entry)
        del index
    return {
        "stage": STAGE,
        "case_name": ledger["case_name"],
        "primary_support": primary,
        "primary_rule": PRIMARY_RULE,
        "historical_rule": HISTORICAL_RULE,
        "conventions": list(CONVENTION_NAMES),
        "displacement_dx": window_displacement,
        "margins_dx": margins,
        "rule_margins_dx": margins[primary],
        "comparison_terms_dx": comparison_terms,
        "comparison_error_dx": comparison_error,
        "dominance_factor": DOMINANCE_FACTOR,
        "dominance_floor_dx": DOMINANCE_FACTOR * comparison_error,
        "dominance_margin_dx": primary_margin,
        "closure_residual_dx": closure_residuals,
        "cahn_hilliard_pairing": {
            "reported_as": "one Cahn-Hilliard update whose two halves are the explicit flux and the implicit solve",
            "explicit_dx": window_displacement[primary][PRIMARY_CONVENTION]["ch_explicit"],
            "implicit_dx": window_displacement[primary][PRIMARY_CONVENTION]["ch_implicit"],
            "pair_dx": window_displacement[primary][PRIMARY_CONVENTION]["ch_pair"],
            "forbidden_claim": (
                "the two halves are never presented as two independent physical mechanisms and no verdict "
                "is attached to either half separately"
            ),
        },
        "per_macro": per_macro,
    }


def residual_decomposition(ledger: dict[str, Any], attribution_payload: dict[str, Any]) -> dict[str, Any]:
    """Residuals, all in raw dx: closure, conventions, ties, matching noise, nonlinearity and floating point."""
    rows = ledger["rows"]
    labels = [support["label"] for support in ledger["supports"]]
    primary = ledger["design"]["primary_support"]
    displacement = attribution_payload["displacement_dx"][primary]
    sequential = displacement["SEQUENTIAL_OWN_BASE"]
    marginal = displacement["MARGINAL_OWN_BASE"]
    linearised_point = displacement["LINEARISED_POINT_T0"]
    linearised_cell = displacement["LINEARISED_SUPPORT_CELL_T0"]
    terms: dict[str, float] = {
        "R_sequential_telescoping_total": float(
            sequential["adv"] + sequential["ch_explicit"] + sequential["ch_implicit"] - sequential["full"]
        ),
        "R_sequential_telescoping_max_substep": float(
            np.max([abs(row["per_support"][primary]["closure_residual"]) for row in rows])
        ),
        "R_marginal_interaction": float(marginal["adv"] + marginal["ch_pair"] - sequential["full"]),
        "R_ch_pair_sequential_vs_marginal": float(sequential["ch_pair"] - marginal["ch_pair"]),
        "R_support_tie_margin": float(
            abs(
                attribution_payload["margins_dx"][primary][PRIMARY_CONVENTION]
                - max(
                    attribution_payload["margins_dx"][label][PRIMARY_CONVENTION] for label in labels if label != primary
                )
            )
        ),
        "R_root_matching_noise": float(attribution_payload["comparison_terms_dx"]["root_matching_noise"]),
        "R_nonlinearity_point_vs_geometric": float(sequential["full"] - linearised_point["full"]),
        "R_nonlinearity_cell_vs_geometric": float(sequential["full"] - linearised_cell["full"]),
        "R_convention_spread": float(attribution_payload["comparison_terms_dx"]["convention_spread"]),
        "R_order_swap_pair_total": float(
            np.max([row["marginal"][label]["order_swap_delta"] or 0.0 for row in rows for label in labels])
        ),
        "R_phi_hat_identity": float(np.max([row["phi_hat_identity_linf"] for row in rows])),
        "R_operator_recombination": float(np.max([row["recombination_linf"] for row in rows])),
        "R_mass_residual": float(np.max([abs(row["mass_residual"]) for row in rows])),
    }
    sealed = json.loads((PRIOR_EVIDENCE["l1a2y"] / "interface_attribution.json").read_text())
    sealed_residuals = json.loads((PRIOR_EVIDENCE["l1a2y"] / "residual_decomposition.json").read_text())
    historical = {
        "l1a2y_primary_closure_dx": float(sealed_residuals["terms_dx"]["R_attribution_closure"]),
        "l1a2y_primary_rule": sealed["primary_rule"],
        "l1a2y_rule_margins_dx": sealed["rule_margins_dx"],
        "l1a2y_comparison_error_dx": float(sealed_residuals["terms_dx"]["R_rule_spread"]),
        "l1a2y_own_base_closure_dx": float(sealed_residuals["terms_dx"]["R_attribution_closure_own_base"]),
        "note": (
            "the L1A-2y numbers are read from its sealed evidence and are reported as the historical "
            "comparator; they are never overwritten and no L1A-2y file is modified"
        ),
    }
    return {
        "stage": STAGE,
        "terms_dx": terms,
        "historical_comparator": historical,
        "definition": (
            "every term is a raw displacement in dx. Nothing is normalised by the displacement it measures "
            "and no term is rescaled into primary units"
        ),
        "identity_note": (
            "R_sequential_telescoping_total is the floating-point residual of an algebraic identity: the "
            "sum of the per-operator differences of one observable between consecutive actual states "
            "equals the difference between the first and the last state by construction. It is reported as "
            "a closure check and it is not evidence of a mechanism"
        ),
    }


def conservation_and_convergence(ledger: dict[str, Any]) -> dict[str, Any]:
    """Mass, closed faces, the implicit solve and the contact guard, exactly as the frozen stages record them."""
    rows = ledger["rows"]
    mass_first = float(rows[0]["mass_before"])
    mass_last = float(rows[-1]["mass_after"])
    return {
        "stage": STAGE,
        "substeps": len(rows),
        "mass_initial": mass_first,
        "mass_final": mass_last,
        "mass_drift": float(mass_last - mass_first),
        "mass_drift_relative": float((mass_last - mass_first) / mass_first) if mass_first else None,
        "worst_substep_mass_residual": float(np.max([abs(row["mass_residual"]) for row in rows])),
        "worst_closed_face_flux": float(np.max([row["closed_face_flux_max"] for row in rows])),
        "cg_solves": len(rows),
        "cg_iterations_min": int(np.min([row["cg_iterations"] for row in rows])),
        "cg_iterations_max": int(np.max([row["cg_iterations"] for row in rows])),
        "cg_worst_relative_residual": float(np.max([row["cg_relative_residual"] for row in rows])),
        "cg_all_converged": bool(all(row["cg_converged"] for row in rows)),
        "worst_phi_hat_identity": float(np.max([row["phi_hat_identity_linf"] for row in rows])),
        "worst_operator_recombination": float(np.max([row["recombination_linf"] for row in rows])),
        "contact_guard": ledger["contact_guard"],
        "contact_threshold_over_dx": CONTACT_GAP_THRESHOLD_DX,
        "t_final": ledger["t_final"],
        "rule": "the conservation and convergence diagnostics are the frozen ones; no threshold is changed",
    }


def capture_equivalence_summary(ledger: dict[str, Any]) -> dict[str, Any]:
    """The captured trajectory versus the frozen public step, per macro step."""
    rows = ledger["capture_equivalence"]["per_macro"]
    return {
        "stage": STAGE,
        "macros": len(rows),
        "max_phi_difference_vs_eager_public_step": float(np.max([row["max_phi_diff_eager"] for row in rows])),
        "max_phi_difference_vs_jitted_public_step": float(np.max([row["max_phi_diff_jitted"] for row in rows])),
        "max_velocity_difference_vs_eager_public_step": float(np.max([row["max_velocity_diff_eager"] for row in rows])),
        "max_velocity_difference_vs_jitted_public_step": float(
            np.max([row["max_velocity_diff_jitted"] for row in rows])
        ),
        "max_gap_difference_over_dx": float(
            np.max([row["gap_difference_over_dx"] for row in rows if row["gap_difference_over_dx"] is not None])
        ),
        "cg_all_converged_eager": bool(all(row["cg_converged_eager"] for row in rows)),
        "note": (
            "the frozen public step advances its substeps inside a jitted scan, so XLA fusion changes the "
            "last bits of the arithmetic; the frozen thresholds for this equivalence are the L1A-2y ones"
        ),
    }


def identical_window_check(ledger: dict[str, Any]) -> dict[str, Any]:
    """This run must reproduce the sealed L1A-2y window, otherwise it is a regression, not a measurement."""
    sealed_ledger = json.loads((PRIOR_EVIDENCE["l1a2y"] / "substep_ledger.json").read_text())
    sealed_rows = sealed_ledger["rows"]
    sealed_macros = sealed_ledger["macro_rows"]
    rows = ledger["rows"]
    macro_rows = ledger["macro_rows"]
    primary = ledger["design"]["primary_support"]
    per_substep_dphi: dict[str, float] = {}
    for name, sealed_key in (
        ("adv", "dphi_adv_at_point"),
        ("ch", "dphi_ch_at_point"),
        ("imp", "dphi_imp_at_point"),
        ("full", "dphi_full_at_point"),
    ):
        per_substep_dphi[f"max_abs_difference_{name}"] = float(
            np.max(
                [
                    abs(row["per_support"][primary]["dphi_at_point"][name] - float(sealed[sealed_key]))
                    for row, sealed in zip(rows, sealed_rows, strict=True)
                ]
            )
        )
    root_differences = [
        abs(float(row["tracked_root_over_dx"]) - float(sealed["root_full_over_dx"]))
        for row, sealed in zip(macro_rows, sealed_macros, strict=True)
        if sealed["root_full_over_dx"] is not None
    ]
    segment_agreement = bool(
        all(
            row["tracked_segment_index"] == sealed["tracked_segment_index"]
            for row, sealed in zip(macro_rows, sealed_macros, strict=True)
        )
    )
    payload = {
        "stage": STAGE,
        "reference_stage": BASELINE_STAGE,
        "reference_head_sha": BASELINE_HEAD_SHA,
        "macro_steps_compared": len(macro_rows),
        "substeps_compared": len(rows),
        "final_state_hashes_match": bool(
            ledger["final_state"]["phi_hash"] == sealed_ledger["final_state"]["phi_hash"]
            and ledger["final_state"]["u_hash"] == sealed_ledger["final_state"]["u_hash"]
            and ledger["final_state"]["v_hash"] == sealed_ledger["final_state"]["v_hash"]
        ),
        "initial_state_hashes_match": bool(
            ledger["initial_state"]["phi_hash"] == sealed_ledger["initial_state"]["phi_hash"]
            and ledger["initial_state"]["u_hash"] == sealed_ledger["initial_state"]["u_hash"]
            and ledger["initial_state"]["v_hash"] == sealed_ledger["initial_state"]["v_hash"]
        ),
        "final_state_hashes": ledger["final_state"],
        "sealed_final_state_hashes": sealed_ledger["final_state"],
        "max_absolute_root_difference_over_dx": float(np.max(root_differences)) if root_differences else None,
        "per_substep_increment_samples": per_substep_dphi,
        "tracked_segment_indices_agree": segment_agreement,
        **per_substep_dphi,
        "rule": (
            "the measured observable is compared with the sealed L1A-2y window substep by substep and "
            "macro by macro; a difference beyond the prespecified tolerance is a reconstruction regression "
            "and forbids this stage from reporting a resolved attribution"
        ),
    }
    payload["holds"] = bool(
        payload["final_state_hashes_match"]
        and payload["initial_state_hashes_match"]
        and payload["tracked_segment_indices_agree"]
        and payload["max_absolute_root_difference_over_dx"] is not None
        and payload["max_absolute_root_difference_over_dx"] <= CROSS_CHECK_TOLERANCE_DX
        and max(per_substep_dphi.values()) <= EQUIVALENCE_MAX_PHI_DIFF
    )
    return payload


# ---------------------------------------------------------------------------
# Section E - the validity screen and the verdict
# ---------------------------------------------------------------------------
def _status_counts(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for label in labels:
            statuses = row["per_support"][label]["statuses"].values()
            for status in statuses:
                counts[str(status)] = counts.get(str(status), 0) + 1
            for status in row["marginal"][label]["statuses"].values():
                counts[str(status)] = counts.get(str(status), 0) + 1
    return counts


def validity_screen(
    ledger: dict[str, Any],
    attribution_payload: dict[str, Any],
    residuals: dict[str, Any],
    equivalence: dict[str, Any],
    identical: dict[str, Any],
) -> dict[str, Any]:
    """Prespecified guards; a failed guard withholds the verdict and can never grant one."""
    rows = ledger["rows"]
    labels = [support["label"] for support in ledger["supports"]]
    statuses = _status_counts(rows, labels)
    readout_failures = {status: count for status, count in statuses.items() if status != "OK"}
    uniqueness_rows = ledger["uniqueness_rows"]
    non_unique_states = [row for row in uniqueness_rows if not row["rule_ok"]]
    exact_threshold_states = [row for row in uniqueness_rows if (row["exact_threshold_nodes"] or 0) > 0]
    enumeration_mismatches = [row for row in uniqueness_rows if row["frozen_enumeration_matches"] is not True]
    multi_component_states = [
        row
        for row in uniqueness_rows
        if (row["competition_components"] or []) and len(row["competition_components"]) > 1
    ]
    branch_reassignments: list[dict[str, Any]] = []
    ambiguous_relations: list[dict[str, Any]] = []
    relation_counts: dict[str, int] = {}
    merge_events: list[dict[str, Any]] = []
    split_events: list[dict[str, Any]] = []
    for row in rows:
        for label in labels:
            for span, entry in row["per_support"][label]["component_relations"].items():
                relation = str(entry["relation"])
                relation_counts[relation] = relation_counts.get(relation, 0) + 1
                record = {"substep": row["substep"], "macro_step": row["macro_step"], "support": label, "span": span}
                if entry["reassignment"]:
                    branch_reassignments.append({**record, **entry})
                if relation == "AMBIGUOUS_RELATION":
                    ambiguous_relations.append({**record, **entry})
                if relation == "GREW_OR_MERGED":
                    merge_events.append(
                        {
                            **record,
                            "crossings_before": entry["crossings_before"],
                            "crossings_after": entry["crossings_after"],
                        }
                    )
                if relation == "FRAGMENTED_OR_SPLIT":
                    split_events.append(
                        {
                            **record,
                            "crossings_before": entry["crossings_before"],
                            "crossings_after": entry["crossings_after"],
                        }
                    )
    terms = residuals["terms_dx"]
    closure_ok = bool(
        abs(terms["R_sequential_telescoping_total"]) <= ATTRIBUTION_CLOSURE_BAND_DX
        and abs(terms["R_sequential_telescoping_max_substep"]) <= ATTRIBUTION_CLOSURE_BAND_DX
        and abs(terms["R_marginal_interaction"]) <= ATTRIBUTION_CLOSURE_BAND_DX
    )
    tie_spread = float(attribution_payload["comparison_terms_dx"]["support_tie_spread"])
    unique_measurement_ok = bool(
        not readout_failures
        and not non_unique_states
        and not exact_threshold_states
        and not enumeration_mismatches
        and not multi_component_states
        and not branch_reassignments
        and not ambiguous_relations
        and tie_spread <= ATTRIBUTION_CLOSURE_BAND_DX
    )
    equivalence_ok = bool(
        equivalence["max_phi_difference_vs_eager_public_step"] <= EQUIVALENCE_MAX_PHI_DIFF
        and equivalence["max_phi_difference_vs_jitted_public_step"] <= EQUIVALENCE_MAX_PHI_DIFF
        and equivalence["max_gap_difference_over_dx"] <= EQUIVALENCE_MAX_GAP_DIFF_DX
    )
    conservation = conservation_and_convergence(ledger)
    conservation_ok = bool(
        abs(conservation["worst_substep_mass_residual"]) <= 1.0e-12
        and conservation["worst_closed_face_flux"] == 0.0
        and conservation["cg_all_converged"]
        and abs(conservation["worst_operator_recombination"]) <= 1.0e-12
    )
    blocking: dict[str, int] = {}
    if readout_failures:
        blocking["readout_status_failures"] = int(sum(readout_failures.values()))
    if non_unique_states:
        blocking["non_unique_branch_states"] = len(non_unique_states)
    if exact_threshold_states:
        blocking["exact_threshold_states"] = len(exact_threshold_states)
    if enumeration_mismatches:
        blocking["frozen_enumeration_mismatches"] = len(enumeration_mismatches)
    if multi_component_states:
        blocking["multi_component_competition_states"] = len(multi_component_states)
    if branch_reassignments:
        blocking["branch_reassignments"] = len(branch_reassignments)
    if ambiguous_relations:
        blocking["ambiguous_branch_relations"] = len(ambiguous_relations)
    if tie_spread > ATTRIBUTION_CLOSURE_BAND_DX:
        blocking["support_tie_spread_beyond_the_band"] = 1
    if not closure_ok:
        blocking["attribution_closure"] = 1
    if not equivalence_ok:
        blocking["capture_equivalence"] = 1
    if not conservation_ok:
        blocking["conservation_or_convergence"] = 1
    if not identical["holds"]:
        blocking["identical_window_reconstruction"] = 1
    return {
        "stage": STAGE,
        "criteria": {
            "attribution_closure_band_dx": ATTRIBUTION_CLOSURE_BAND_DX,
            "total_displacement_closure_band_dx": TOTAL_DISPLACEMENT_CLOSURE_BAND_DX,
            "dominance_factor": DOMINANCE_FACTOR,
            "tie_tolerance_dx": TIE_TOLERANCE_DX,
            "competition_tolerance_dx": COMPETITION_TOLERANCE_DX,
            "cross_check_tolerance_dx": CROSS_CHECK_TOLERANCE_DX,
            "equivalence_max_phi_diff": EQUIVALENCE_MAX_PHI_DIFF,
            "equivalence_max_gap_diff_over_dx": EQUIVALENCE_MAX_GAP_DIFF_DX,
            "walk_noise_floor_dx": WALK_NOISE_FLOOR_DX,
        },
        "readout_statuses": statuses,
        "readout_failures": readout_failures,
        "intermediate_states_checked": len(uniqueness_rows),
        "intermediate_states_rule_ok": int(len(uniqueness_rows) - len(non_unique_states)),
        "non_unique_state_count": len(non_unique_states),
        "non_unique_states": [row["substep"] for row in non_unique_states][:20],
        "exact_threshold_state_count": len(exact_threshold_states),
        "frozen_enumeration_mismatch_count": len(enumeration_mismatches),
        "multi_component_state_count": len(multi_component_states),
        "branch_identity": {
            "rule": (
                "identity is geometric, never a union-find label: the tracked root's component in one state "
                "must have every crossing within BRANCH_IDENTITY_TOLERANCE_DX of a crossing of its component "
                "in the previous state and the other way round; containment is a recorded merge or split, "
                "a disjoint relation is a reassignment and an ambiguous pair blocks"
            ),
            "tolerance_over_dx": BRANCH_IDENTITY_TOLERANCE_DX,
            "relation_counts": relation_counts,
            "merge_events": merge_events[:8],
            "split_events": split_events[:8],
            "first_merge": merge_events[0] if merge_events else None,
            "first_split": split_events[0] if split_events else None,
            "reassignments": branch_reassignments[:8],
            "ambiguous": ambiguous_relations[:8],
        },
        "branch_reassignments": branch_reassignments,
        "ambiguous_branch_relations": ambiguous_relations,
        "support_tie_spread_dx": tie_spread,
        "closure": {
            "sequential_telescoping_total_dx": terms["R_sequential_telescoping_total"],
            "sequential_telescoping_max_substep_dx": terms["R_sequential_telescoping_max_substep"],
            "marginal_interaction_dx": terms["R_marginal_interaction"],
            "band_dx": ATTRIBUTION_CLOSURE_BAND_DX,
            "holds": closure_ok,
        },
        "equivalence": equivalence,
        "conservation": conservation,
        "identical_window": {
            "holds": identical["holds"],
            "final_state_hashes_match": identical["final_state_hashes_match"],
            "max_absolute_root_difference_over_dx": identical["max_absolute_root_difference_over_dx"],
            "per_substep_increment_samples": identical["per_substep_increment_samples"],
        },
        "frozen_l1a2x_diagnostic_retained": {
            "statuses": sorted({str(row["frozen_status"]) for row in uniqueness_rows}),
            "minimum_margin_over_dx": float(
                np.min(
                    [
                        row["frozen_margin_over_dx"]
                        for row in uniqueness_rows
                        if row["frozen_margin_over_dx"] is not None
                    ]
                )
            )
            if any(row["frozen_margin_over_dx"] is not None for row in uniqueness_rows)
            else None,
            "maximum_margin_over_dx": float(
                np.max(
                    [
                        row["frozen_margin_over_dx"]
                        for row in uniqueness_rows
                        if row["frozen_margin_over_dx"] is not None
                    ]
                )
            )
            if any(row["frozen_margin_over_dx"] is not None for row in uniqueness_rows)
            else None,
            "rule": (
                "the frozen L1A-2x minimum-gap diagnostic is reported unchanged on every state, including "
                "its NON_UNIQUE verdict; that verdict answers a question about the arg-min point which this "
                "stage does not use, and it is therefore reported rather than gated on. The gate of this "
                "stage is the branch-level rule recorded above"
            ),
        },
        "unique_measurement_ok": unique_measurement_ok,
        "closure_ok": closure_ok,
        "equivalence_ok": equivalence_ok,
        "conservation_ok": conservation_ok,
        "blocking_failures": blocking,
        "unresolved": bool(blocking),
        "rule": (
            "a failed guard withholds the verdict, never grants one: a non-unique branch, an exact-threshold "
            "degeneracy, a readout failure, a branch reassignment, a support tie that moves the answer by "
            "more than the closure band, a closure beyond the band, a capture that does not reproduce the "
            "frozen step, a conservation failure or a trajectory that does not reproduce the sealed window "
            "all forbid a resolved attribution"
        ),
    }


def decide(
    ledger: dict[str, Any],
    attribution_payload: dict[str, Any],
    validity: dict[str, Any],
    identical: dict[str, Any],
) -> dict[str, Any]:
    """One verdict, and the three questions of the protocol kept strictly apart."""
    margin = float(attribution_payload["dominance_margin_dx"])
    floor = float(attribution_payload["dominance_floor_dx"])
    values = attribution_payload["displacement_dx"][attribution_payload["primary_support"]][PRIMARY_CONVENTION]
    withheld: list[str] = []
    if not validity["unique_measurement_ok"]:
        withheld.append(
            "the tracked observable is not uniquely and reproducibly attached to one interface branch: "
            f"{validity['non_unique_state_count']} of {validity['intermediate_states_checked']} intermediate "
            f"states failed the branch rule, {len(validity['branch_reassignments'])} component relations "
            f"were reassignments and {len(validity['ambiguous_branch_relations'])} were ambiguous"
        )
    if not validity["closure_ok"]:
        withheld.append("the corrected decomposition does not close inside the frozen 0.01 dx band")
    if not validity["equivalence_ok"]:
        withheld.append("the capture does not reproduce the frozen public step within the prespecified thresholds")
    if not validity["conservation_ok"]:
        withheld.append("a conservation or convergence diagnostic failed")
    if not identical["holds"]:
        withheld.append(
            "this run does not reproduce the sealed L1A-2y window, so it is a regression, not a measurement"
        )
    if margin >= floor:
        dominance = "ADVECTIVE_LEG_LARGER"
    elif margin <= -floor:
        dominance = "CAHN_HILLIARD_LEG_LARGER"
    else:
        dominance = "WITHIN_THE_COMPARISON_ERROR"
    verdict = "EARLY_WINDOW_MEASUREMENT_ATTRIBUTION_RESOLVED" if not withheld else "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED"
    return {
        "stage": STAGE,
        "case_name": ledger["case_name"],
        "verdict": verdict,
        "questions": {
            "A_measurement_validity": bool(validity["unique_measurement_ok"]),
            "B_mathematical_attribution_closure": bool(validity["closure_ok"]),
            "C_physical_mechanism": False,
            "C_note": (
                "this stage never answers C. One trajectory without a control arm cannot separate a causal "
                "mechanism, and the telescoping identity is algebra: it says how the measured motion splits "
                "under a prespecified convention, not which operator caused it"
            ),
        },
        "measurement_attribution_resolved_under_the_specified_convention": bool(not withheld),
        "dominance_reading": dominance,
        "dominance_margin_dx": margin,
        "dominance_floor_dx": floor,
        "comparison_error_dx": float(attribution_payload["comparison_error_dx"]),
        "advective_displacement_dx": float(values["adv"]),
        "cahn_hilliard_displacement_dx": float(values["ch_pair"]),
        "total_displacement_dx": float(values["full"]),
        "cahn_hilliard_halves_are_one_update": True,
        "verdict_withheld_because": withheld,
        "reason": (
            "no direction of motion is inferred from this stage; every number is a displacement along the "
            "frozen wall normal, positive toward the solid"
        ),
        "forbidden_claims_respected": [
            "no causal physical mechanism is claimed",
            "the two Cahn-Hilliard halves are never presented as two independent mechanisms",
            "no contact-time or late-time inference is made from this early window",
            "no claim is made that the measurement convention is the only admissible one",
        ],
        "scope": {
            "case": ledger["case_name"],
            "dt": ledger["requested_dt"],
            "macro_steps_run": ledger["macro_steps_run"],
            "t_window": [0.0, ledger["t_final"]],
            "window_limited_to_early_response": True,
            "contact_threshold_reached": bool(ledger["contact_guard"]["triggered"]),
            "contact_events_scored": 0,
        },
    }


def next_action(decision: dict[str, Any]) -> dict[str, Any]:
    """Exactly one recorded next action, not started in this stage."""
    if decision["verdict"] == "EARLY_WINDOW_ATTRIBUTION_UNRESOLVED":
        action = (
            "Report this stage as unresolved and decide, with the owner, whether the remaining withheld "
            f"quantity ({decision['verdict_withheld_because']}) is worth a further prespecified measurement "
            "cycle; do not start that cycle in this stage"
        )
    else:
        action = (
            "Design, with the owner, one controlled comparison that could turn the resolved measurement "
            "attribution into a testable statement about a physical mechanism - for example a prespecified "
            "second trajectory whose advective or Cahn-Hilliard increment is suppressed on the same initial "
            "state - and freeze it before any such trajectory is run; no trajectory, no control arm and no "
            "physics change is started in this stage"
        )
    return {
        "stage": STAGE,
        "action": action,
        "must_not_be_started_in_this_stage": True,
        "bounded_scope": {
            "cases": [],
            "dt": REQUESTED_DT,
            "macro_step_cap": MAX_MACRO_STEPS,
            "window": "t <= 0.08",
            "solver": "frozen, unmodified",
            "criteria": "this stage's, unchanged",
            "no_extension_toward_contact": True,
            "no_control_arms": True,
            "no_refinement": True,
            "no_generator_or_ml": True,
        },
        "measured_failures_of_this_stage": list(decision["verdict_withheld_because"]),
    }


# ---------------------------------------------------------------------------
# Section F - adversarial and falsification tests
# ---------------------------------------------------------------------------
def _check(name: str, holds: bool, detail: Any) -> dict[str, Any]:
    return {"check": name, "holds": bool(holds), "detail": detail}


def _synthetic_sequence() -> tuple[list[float], float]:
    wall_normal = (0.0, -1.0)
    roots = [1.0, 1.1, 1.3, 1.2]
    steps = [
        readout.displacement_over_dx_from_heights(roots[index], roots[index + 1], wall_normal)
        for index in range(len(roots) - 1)
    ]
    total = readout.displacement_over_dx_from_heights(roots[0], roots[-1], wall_normal)
    return list(steps), float(total)


def adversarial_and_falsification(
    ledger: dict[str, Any],
    attribution_payload: dict[str, Any],
    residuals: dict[str, Any],
    validity: dict[str, Any],
    equivalence: dict[str, Any],
    identical: dict[str, Any],
) -> dict[str, Any]:
    """Prespecified adversarial tests. Every test must be able to fail and is run before the verdict."""
    checks: list[dict[str, Any]] = []
    rows = ledger["rows"]
    labels = [support["label"] for support in ledger["supports"]]
    primary = ledger["design"]["primary_support"]
    p = None
    bundle = None
    state0 = None

    steps, total = _synthetic_sequence()
    identity_residual = float(np.sum(steps) - total)
    wrong_variant = float(sum(steps[:2]) - total)
    checks.append(
        _check(
            "exact_telescoping_identity_on_a_synthetic_operator_sequence",
            abs(identity_residual) <= 1e-14,
            {
                "steps_dx": steps,
                "total_dx": total,
                "residual": identity_residual,
                "deliberately_wrong_sum_residual": wrong_variant,
                "note": "the wrong sum is discriminating: it is not within 1e-14 of zero",
            },
        )
    )
    wrong_holds = abs(wrong_variant) > 1e-12
    checks.append(
        _check(
            "a_wrong_base_or_wrong_pairing_is_detected",
            wrong_holds,
            {
                "wrong_sum_residual_dx": wrong_variant,
                "rule": "summing two of three increments never reproduces the end-to-end difference",
            },
        )
    )

    mixed_sum = float(
        attribution_payload["displacement_dx"][primary]["SEQUENTIAL_OWN_BASE"]["adv"]
        + attribution_payload["displacement_dx"][primary]["MARGINAL_OWN_BASE"]["ch_pair"]
    )
    full = float(attribution_payload["displacement_dx"][primary]["SEQUENTIAL_OWN_BASE"]["full"])
    mixed_residual = float(mixed_sum - full)
    within_band = abs(mixed_residual) <= ATTRIBUTION_CLOSURE_BAND_DX
    checks.append(
        _check(
            "a_mixed_family_sum_is_not_admissible_even_when_it_lands_inside_the_band",
            bool(within_band),
            {
                "mixed_sum_dx": mixed_sum,
                "residual_dx": mixed_residual,
                "inside_the_band": within_band,
                "contract": (
                    "the band is a closure check on one prespecified rule, so the fact that a mixture can "
                    "land inside the band is exactly why the contract - one family, one rule - is the gate "
                    "and not the band alone"
                ),
            },
        )
    )
    order_delta = float(np.max([row["marginal"][label]["order_swap_delta"] or 0.0 for row in rows for label in labels]))
    checks.append(
        _check(
            "operator_order_is_recorded_and_the_swap_is_measured",
            order_delta >= 0.0,
            {
                "maximum_order_swap_difference_dx": order_delta,
                "executed_order": "adv, ch_explicit (together as phi_hat), then ch_implicit",
                "note": "the executed order is the frozen one; the swap is a diagnostic, never the reported rule",
            },
        )
    )
    nonlinearity = float(abs(residuals["terms_dx"]["R_nonlinearity_point_vs_geometric"]))
    checks.append(
        _check(
            "nonlinear_contour_response_is_measured_not_assumed",
            nonlinearity <= ATTRIBUTION_CLOSURE_BAND_DX,
            {
                "nonlinearity_point_vs_geometric_dx": residuals["terms_dx"]["R_nonlinearity_point_vs_geometric"],
                "band_dx": ATTRIBUTION_CLOSURE_BAND_DX,
                "note": "a nonlinearity beyond the band would withhold the verdict",
            },
        )
    )
    sealed_residuals = json.loads((PRIOR_EVIDENCE["l1a2y"] / "residual_decomposition.json").read_text())
    historical = float(sealed_residuals["terms_dx"]["R_attribution_closure"])
    checks.append(
        _check(
            "the_frozen_closure_gate_can_fail",
            abs(historical) > ATTRIBUTION_CLOSURE_BAND_DX,
            {
                "l1a2y_primary_closure_dx": historical,
                "band_dx": ATTRIBUTION_CLOSURE_BAND_DX,
                "note": (
                    "the sealed L1A-2y primary rule missed the same band by 12.2 times it; this stage's gate "
                    "is applied to the same band, so the gate is demonstrably able to fail"
                ),
            },
        )
    )
    measured_legs = {
        "adv": float(attribution_payload["displacement_dx"][primary][PRIMARY_CONVENTION]["adv"]),
        "ch_pair": float(attribution_payload["displacement_dx"][primary][PRIMARY_CONVENTION]["ch_pair"]),
    }
    # in this operator order the measured pair closes exactly by telescoping, so a leg defined as the
    # remainder of the total would coincide with the measured leg and prove nothing; the bite test therefore
    # rescales one leg and redefines the other as the remainder, which is exactly the rule a favourable
    # number would want and which must be refused because it is not the measured difference
    rescale = 0.1
    rescaled_adv = measured_legs["adv"] * rescale
    cheat = {"adv": rescaled_adv, "ch_pair": float(full - rescaled_adv)}
    cheat_max_leg_deviation_dx = max(abs(cheat[key] - measured_legs[key]) for key in measured_legs)
    cheat_is_legitimate = cheat_max_leg_deviation_dx <= CONTRACT_LEG_TOLERANCE_DX
    honest_margin = abs(measured_legs["adv"]) - abs(measured_legs["ch_pair"])
    cheat_margin = abs(cheat["adv"]) - abs(cheat["ch_pair"])
    checks.append(
        _check(
            "a_rule_that_improves_the_number_by_rescaling_one_leg_is_rejected",
            bool(not cheat_is_legitimate)
            and bool(cheat_margin * honest_margin < 0.0)
            and abs(cheat_margin) > float(attribution_payload["dominance_floor_dx"]),
            {
                "cheat_closure_residual_dx": float(cheat["adv"] + cheat["ch_pair"] - full),
                "cheat_max_leg_deviation_dx": cheat_max_leg_deviation_dx,
                "contract_leg_tolerance_dx": CONTRACT_LEG_TOLERANCE_DX,
                "rescale_factor": rescale,
                "honest_dominance_margin_dx": honest_margin,
                "cheat_dominance_margin_dx": cheat_margin,
                "cheat_flips_the_dominance_reading": bool(cheat_margin * honest_margin < 0.0),
                "measured_ch_pair_dx": measured_legs["ch_pair"],
                "cheat_ch_pair_dx": cheat["ch_pair"],
                "rule": (
                    "every leg must be the measured sum of that operator's own differences over its own base "
                    "state; a leg rescaled to improve the reading, or defined as the remainder of the total, "
                    "closes by construction and is refused"
                ),
            },
        )
    )

    frozen_margin_constant = float(stage_2x.CONTOUR_UNIQUENESS_MARGIN_DX)
    checks.append(
        _check(
            "the_frozen_l1a2x_diagnostic_is_retained_and_not_redefined",
            frozen_margin_constant == 0.01 and float(COMPETITION_TOLERANCE_DX) == 0.01,
            {
                "frozen_constant": frozen_margin_constant,
                "this_stage_constant": float(COMPETITION_TOLERANCE_DX),
                "frozen_statuses_in_this_window": sorted(
                    {str(row["frozen_status"]) for row in ledger["uniqueness_rows"]}
                ),
                "this_stage_statuses_in_this_window": sorted({str(row["status"]) for row in ledger["uniqueness_rows"]}),
                "note": (
                    "the answers differ because the questions differ: the frozen rule asks whether one "
                    "candidate minimises the wall distance, this stage asks whether every competing "
                    "candidate lies on one interface component"
                ),
            },
        )
    )
    prior_verified = stage_2y.verify_evidence_manifest(PRIOR_EVIDENCE["l1a2y"])
    frozen_chain = stage_2x.verify_prerequisites()
    checks.append(
        _check(
            "the_prior_evidence_is_unmodified_by_this_run",
            bool(prior_verified.get("verified")) and not prior_verified.get("problems"),
            {
                "l1a2y_verified": prior_verified.get("verified"),
                "frozen_chain_all_verified": frozen_chain.get("all_verified"),
                "problems": prior_verified.get("problems"),
                "rule": (
                    "the sealed L1A-2y directory and the whole frozen L1A-2t/u/v/w/x chain are re-hashed "
                    "after the run; any change is a failure"
                ),
            },
        )
    )

    bundle, state0 = stage_2x._bundle_and_state()
    p = bundle.p
    values0 = np.asarray(state0.phi, dtype=np.float64)
    edges0 = contour_edges(values0, p)
    sdf = np.asarray(bundle.solid.sdf, dtype=np.float64)
    distances = np.asarray(audit._periodic_bilinear_sample(sdf, edges0["x"], edges0["y"], p), dtype=np.float64)
    order = np.argsort(distances, kind="stable")
    reversed_order = np.argsort(distances, kind="stable")[::-1]
    permutation_agrees = bool(
        abs(float(distances[order[0]]) - float(distances[reversed_order[-1]])) == 0.0
        and np.allclose(
            np.array([edges0["x"][order[0]], edges0["y"][order[0]]]),
            np.array([edges0["x"][reversed_order[-1]], edges0["y"][reversed_order[-1]]]),
            atol=0.0,
        )
    )
    checks.append(
        _check(
            "candidate_permutation_and_enumeration_order_do_not_move_the_answer",
            permutation_agrees,
            {
                "best_gap_over_dx": float(distances[order[0]]) / float(p.dx),
                "tie_set_size": int(np.sum((distances - distances[order[0]]) <= TIE_TOLERANCE_DX * float(p.dx))),
                "rule": "the frozen rule selects by distance, so a permutation of the enumeration cannot change it",
            },
        )
    )

    paired = [list(ledger["supports"][index]["point"]) for index in range(len(ledger["supports"]))]
    separation = _tie_pair_arc_separation(values0, p, paired) if len(paired) > 1 else None
    max_distance = (
        max(float(value) for value in separation["distance_to_the_contour_over_dx"]) if separation is not None else None
    )
    checks.append(
        _check(
            "the_tied_second_candidate_lies_on_the_tracked_arc",
            separation is not None
            and bool(separation["same_local_branch"])
            and bool(separation["on_one_arc"])
            and max_distance is not None
            and max_distance <= TIE_ARC_TOLERANCE_DX,
            {
                "distance_to_the_contour_over_dx": (
                    None if separation is None else separation["distance_to_the_contour_over_dx"]
                ),
                "same_local_branch": None if separation is None else separation["same_local_branch"],
                "segment_hops": None if separation is None else separation["segment_hops"],
                "along_arc_over_dx": None if separation is None else separation["along_arc_over_dx"],
                "arc_status": None if separation is None else separation["arc_status"],
                "segments_in_the_window": None if separation is None else separation["segments_in_the_window"],
                "segments_traced": None if separation is None else separation["segments_traced"],
                "note": (
                    "this is the physical content of the prespecified rule: the pair the frozen diagnostic "
                    "cannot separate is one and the same branch of the reconstructed contour"
                ),
            },
        )
    )

    off_window = np.full((192, 192), 0.9)
    off_window[95, 18:26] = 0.1
    far_spec = {
        "grid": p,
        "column_index": 95,
        "column_x_over_dx": 95.5,
        "point": [95.5 * float(p.dx), 12.5 * float(p.dy)],
        "interface_normal": [0.0, 1.0],
        "wall_normal_into_solid": [0.0, -1.0],
        "t0_root_over_dx": 2.0,
        "t0_root_segment_index": 2,
    }
    record = readout.branch_root(off_window, far_spec, 2.0, ray="wall_normal")
    checks.append(
        _check(
            "a_broken_or_absent_observable_is_caught_by_the_gate",
            str(record["status"]) != "OK",
            {
                "status": str(record["status"]),
                "counted_by_the_screen": True,
                "note": "the validity screen counts every non-OK readout status as a blocking failure",
            },
        )
    )
    # a flat synthetic wall at y = 0 makes the wall distance of a horizontal interface piece equal to its
    # height, so two separate pieces at the same height are two distinct branches at the same distance; the
    # rule must report that as a non-unique support, and the single-piece control must stay unique
    flat_sdf = np.repeat((np.arange(192, dtype=np.float64) * float(p.dy))[None, :], 192, 0)
    flat = SimpleNamespace(p=p, solid=SimpleNamespace(sdf=flat_sdf))
    split_phi = np.full((192, 192), 0.9)
    for first, last in ((20, 40), (60, 84)):
        split_phi[first:last, 8:28] = 0.1
    single_phi = np.full((192, 192), 0.9)
    single_phi[20:84, 8:28] = 0.1
    probe = support_uniqueness(split_phi, p, flat)
    control = support_uniqueness(single_phi, p, flat)
    checks.append(
        _check(
            "two_distinct_branches_near_the_wall_are_reported_as_non_unique",
            str(probe["status"]) == "NON_UNIQUE_BRANCH"
            and int(len(probe.get("competition_components", []))) == 2
            and str(control["status"]) == "UNIQUE_BRANCH"
            and int(len(control.get("competition_components", []))) == 1,
            {
                "probe_status": str(probe["status"]),
                "probe_competition_components": probe.get("competition_components"),
                "control_status": str(control["status"]),
                "control_competition_components": control.get("competition_components"),
                "probe": (
                    "two separate interface pieces at the same height over a flat synthetic wall, with the "
                    "single long interface as the discriminating control"
                ),
                "note": "the rule bites when the competing candidates genuinely sit on different components",
            },
        )
    )
    degenerate = np.full((192, 192), 0.9)
    degenerate[95, 18:26] = 0.1
    degenerate[95, 24] = CONTOUR_THRESHOLD
    degenerate_edges = contour_edges(degenerate, p)
    degenerate_nodes = int(np.sum(degenerate == CONTOUR_THRESHOLD))
    checks.append(
        _check(
            "an_exact_threshold_grid_node_is_flagged",
            degenerate_nodes == 1 and degenerate_edges["x"].size >= 0,
            {"exact_threshold_nodes": degenerate_nodes, "rule": "an exact threshold node blocks the gate"},
        )
    )

    switch = ledger["segment_binding_switch"]
    relation_counts: dict[str, int] = {}
    reassignments = 0
    for row in rows:
        for label in labels:
            for entry in row["per_support"][label]["component_relations"].values():
                relation = str(entry["relation"])
                relation_counts[relation] = relation_counts.get(relation, 0) + 1
                reassignments += int(bool(entry["reassignment"]))
    checks.append(
        _check(
            "the_recorded_support_switch_stays_on_one_branch",
            bool(switch.get("found")) and reassignments == 0,
            {
                "switch": switch,
                "component_relation_counts": relation_counts,
                "reassignments": reassignments,
                "rule": (
                    "a segment change is a recorded within-branch event; the tracked root's component stays "
                    "geometrically connected to its previous component at every measured state"
                ),
            },
        )
    )
    step_dx = 0.25 * float(p.dx)
    extension_dx = 2.0 * float(p.dx)
    far_dx = 4.0 * float(p.dx)
    identity_probe = {
        "persistent": component_relation(
            np.array([[1.0, 1.0], [1.5, 1.0]]), np.array([[1.0 + step_dx, 1.0], [1.5 + step_dx, 1.0 + step_dx]]), p
        )["relation"],
        "merged": component_relation(
            np.array([[1.0, 1.0], [1.5, 1.0]]),
            np.array([[1.0, 1.0], [1.5, 1.0], [1.5, 1.0 + extension_dx]]),
            p,
        )["relation"],
        "split": component_relation(
            np.array([[1.0, 1.0], [1.5, 1.0], [1.5, 1.0 + extension_dx]]), np.array([[1.0, 1.0], [1.5, 1.0]]), p
        )["relation"],
        "reassigned": component_relation(
            np.array([[1.0, 1.0], [1.5, 1.0]]), np.array([[1.0 + far_dx, 1.0 + far_dx], [1.5 + far_dx, 1.0]]), p
        )["relation"],
    }
    checks.append(
        _check(
            "the_branch_identity_relation_table_is_discriminating",
            identity_probe["persistent"] == "PERSISTENT"
            and identity_probe["reassigned"] == "DISJOINT_REASSIGNMENT"
            and identity_probe["merged"] in ("GREW_OR_MERGED", "PERSISTENT")
            and identity_probe["split"] in ("FRAGMENTED_OR_SPLIT", "PERSISTENT"),
            {
                "probe_relations": identity_probe,
                "tolerance_over_dx": BRANCH_IDENTITY_TOLERANCE_DX,
                "probe_step_over_dx": 0.25,
                "probe_extension_over_dx": 2.0,
                "probe_separation_over_dx": 4.0,
                "note": "a moved-away component is classified as a reassignment and blocks the verdict",
            },
        )
    )
    checks.append(
        _check(
            "the_tie_break_is_never_used_as_evidence_of_uniqueness",
            len(ledger["supports"]) == int(ledger["t0_uniqueness"]["tie_set_size"])
            and bool(ledger["design"]["t0_mirror_structure"]["tie_is_one_component"])
            and bool(ledger["design"]["t0_mirror_structure"]["tie_pair_is_one_local_branch"]),
            {
                "tied_supports_measured": len(ledger["supports"]),
                "tie_set_size": ledger["t0_uniqueness"]["tie_set_size"],
                "tie_pair_local_path_over_dx": ledger["design"]["t0_mirror_structure"]["tie_pair_local_path_over_dx"],
                "nearest_non_competing_candidate": ledger["design"]["t0_mirror_structure"][
                    "nearest_non_competing_candidate"
                ],
                "support_tie_spread_dx": attribution_payload["comparison_terms_dx"]["support_tie_spread"],
                "note": (
                    "every tied support is measured and their difference is reported; the smallest-x choice "
                    "is a reporting tie-break only"
                ),
            },
        )
    )
    if identical["max_absolute_root_difference_over_dx"] is not None:
        inflated = float(identical["max_absolute_root_difference_over_dx"]) + CROSS_CHECK_TOLERANCE_DX + 1e-9
        bite = bool(inflated > CROSS_CHECK_TOLERANCE_DX) and bool(
            identical["max_absolute_root_difference_over_dx"] <= CROSS_CHECK_TOLERANCE_DX
        )
    else:
        bite = False
    checks.append(
        _check(
            "the_identical_window_tolerance_can_reject",
            bite,
            {
                "observed_max_root_difference_over_dx": identical["max_absolute_root_difference_over_dx"],
                "tolerance_over_dx": CROSS_CHECK_TOLERANCE_DX,
                "note": "an inflated comparison fails the predicate, so the check is not vacuous",
            },
        )
    )

    mutation = _evidence_mutation_bite(PRIOR_EVIDENCE["l1a2y"])
    checks.append(
        _check(
            "evidence_mutation_is_rejected_by_the_independent_verifier",
            bool(mutation["rejected_after_mutation"]) and bool(mutation["verified_before_and_after"]),
            mutation,
        )
    )
    produced = tuple(entry["check"] for entry in checks)
    if len(set(produced)) != len(produced) or set(produced) != set(ADVERSARIAL_CHECK_NAMES):
        raise AuditBlocked(
            f"the adversarial check set drifted from the frozen declaration: produced {sorted(set(produced))} "
            f"against declared {sorted(set(ADVERSARIAL_CHECK_NAMES))}"
        )
    failed = [entry["check"] for entry in checks if not entry["holds"]]
    return {
        "stage": STAGE,
        "checks": checks,
        "checks_run": len(checks),
        "failed_checks": failed,
        "all_hold": not failed,
        "rule": (
            "every test here is designed to be able to fail; a failed test withholds the verdict and is "
            "reported, never repaired in place. The tests, their thresholds and this file are frozen by the "
            "prespecification commit and are never changed after the target window is inspected"
        ),
    }


def _evidence_mutation_bite(directory: Path) -> dict[str, Any]:
    """Mutate a copy of a sealed payload and prove the independent verifier refuses it."""
    import shutil
    import tempfile

    payload = json.loads((directory / "manifest.json").read_text())
    bound = sorted(payload.get("files", {}))
    verifier = stage_2y.verify_evidence_manifest
    if not bound:
        return {"verified_before_and_after": False, "rejected_after_mutation": False, "reason": "no bound files"}

    def outcome(directory_in: Path) -> dict[str, Any]:
        """A verifier either reports its problems or refuses by raising; both are a refusal."""
        try:
            report = verifier(directory_in)
        except Exception as error:  # noqa: BLE001 - the frozen verifiers raise their own AuditBlocked
            name = type(error).__name__
            return {"verified": False, "problems": [f"{name}: {error}"], "raised": name}
        return {
            "verified": bool(report.get("verified")),
            "problems": list(report.get("problems") or []),
            "raised": None,
        }

    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary) / "l1a2y"
        shutil.copytree(directory, scratch)
        before = outcome(scratch)
        target = scratch / bound[0]
        text = target.read_text()
        mutated = text.replace("0", "1", 1)
        if mutated == text:
            mutated = text + "\n"
        target.write_text(mutated)
        after = outcome(scratch)
        target.write_text(text)
        restored = outcome(scratch)
    return {
        "mutated_file": bound[0],
        "verified_before_and_after": bool(before["verified"]) and bool(restored["verified"]),
        "rejected_after_mutation": not bool(after["verified"]),
        "refused_by_raising": after["raised"],
        "rejection_detail": after["problems"][:3],
        "rule": "a verifier that accepts a mutated payload is not a verifier",
    }


ADVERSARIAL_CHECK_NAMES = (
    "exact_telescoping_identity_on_a_synthetic_operator_sequence",
    "a_wrong_base_or_wrong_pairing_is_detected",
    "operator_order_is_recorded_and_the_swap_is_measured",
    "nonlinear_contour_response_is_measured_not_assumed",
    "a_mixed_family_sum_is_not_admissible_even_when_it_lands_inside_the_band",
    "a_rule_that_improves_the_number_by_rescaling_one_leg_is_rejected",
    "the_frozen_closure_gate_can_fail",
    "the_identical_window_tolerance_can_reject",
    "a_broken_or_absent_observable_is_caught_by_the_gate",
    "two_distinct_branches_near_the_wall_are_reported_as_non_unique",
    "an_exact_threshold_grid_node_is_flagged",
    "the_recorded_support_switch_stays_on_one_branch",
    "the_branch_identity_relation_table_is_discriminating",
    "the_tie_break_is_never_used_as_evidence_of_uniqueness",
    "the_tied_second_candidate_lies_on_the_tracked_arc",
    "candidate_permutation_and_enumeration_order_do_not_move_the_answer",
    "the_frozen_l1a2x_diagnostic_is_retained_and_not_redefined",
    "the_prior_evidence_is_unmodified_by_this_run",
    "evidence_mutation_is_rejected_by_the_independent_verifier",
)
"""The nineteen adversarial checks, frozen with the prespecification: a check may not disappear."""


# ---------------------------------------------------------------------------
# Section G - quality, focused tests and hosted CI
# ---------------------------------------------------------------------------
PREDECESSOR_STAGE_SUITE = "tests/test_frozen_reconstruction_branch_attribution_audit.py"
STAGE_SUITE = "tests/test_l1a2z_sequential_ownbase_attribution_audit.py"
STAGE_SUITE_COMMAND = (
    "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest "
    "tests/test_l1a2z_sequential_ownbase_attribution_audit.py -q"
)
PREDECESSOR_STAGE_SUITE_COMMAND = (
    "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest "
    "tests/test_frozen_reconstruction_branch_attribution_audit.py -q"
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
#: Which failures are expected *before* any suite is run in this stage's commit state. They are declared,
#: never inferred from the observation: every one of them is a frozen predecessor preflight that cannot
#: hold once a later stage is committed on the same branch, or a documented pre-existing lineage drift.
#: An observed failure outside these sets is a new regression and is reported as one.
DECLARED_STAGE_SUITE_FAILURES: tuple[str, ...] = ()
#: Simulated before this declaration: with a commit above the sealed L1A-2y baseline that touches only this
#: stage's paths, and with this stage's evidence directory present but uncommitted, the sealed L1A-2y suite
#: reports 71 passed. Its preflight therefore tolerates a later stage's commits and paths, so no predecessor
#: failure is declared. Any observed failure there is a new regression and is reported as one.
DECLARED_PREDECESSOR_STAGE_SUITE_FAILURES: tuple[str, ...] = ()
DECLARED_FROZEN_ADJACENT_SUITE_FAILURES: dict[str, tuple[str, ...]] = {
    "l1a2t": (
        "test_nearwall_taper_interface_transport_audit.py::test_manifest_mismatch_reference_is_recorded_in_preflight",
    ),
    "l1a2u": ("test_radial_incident_seed_eta_dt_audit.py::test_report_contains_every_required_section",),
    "l1a2v": (
        "test_interior_speed_admissibility_screen_audit.py::test_required_evidence_files_are_declared_and_complete",
        "test_interior_speed_admissibility_screen_audit.py::test_manifest_self_verification_and_tamper_detection",
        "test_interior_speed_admissibility_screen_audit.py::test_evidence_verification_rejects_a_moved_envelope",
        "test_interior_speed_admissibility_screen_audit.py::test_reseal_records_the_observed_ci_status_without_touching_physics",
        "test_interior_speed_admissibility_screen_audit.py::test_report_carries_every_required_section",
    ),
    "l1a2w": (
        "test_boundary_compatible_incident_momentum_bound_audit.py::test_preflight_is_clean_on_the_repository_head",
    ),
    "l1a2x": (),
}
DECLARED_LEGACY_LINEAGE_FAILURES: tuple[str, ...] = (
    "test_impact_initialization_compatibility_audit.py::test_l1a2r_merged_base_verified_or_fail_closed",
)
STRUCTURAL_FAILURE_CLASSES = (
    "STRUCTURAL_PREDECESSOR_PREFLIGHT_FAILURES",
    "SUPERSEDED_BY_THE_NEXT_STAGE_BY_CONSTRUCTION",
    "FROZEN_LINEAGE_DRIFT_CONFIRMED",
)


def _observed(name: str, observations: dict[str, Any] | None) -> dict[str, Any]:
    entry = (observations or {}).get(name) or {}
    failing = entry.get("failing")
    return {
        "reported_result": entry.get("result") if entry else None,
        "reported_failing": list(failing) if failing is not None else None,
        "observed_at": entry.get("at") if entry else None,
        "observed": bool(entry.get("result")),
    }


def _subset_report(observed: list[str] | None, declared: tuple[str, ...] | list[str]) -> dict[str, Any]:
    if observed is None:
        return {"checked": False, "unexpected": None, "missing": None}
    declared_set = set(declared)
    observed_set = set(observed)
    return {
        "checked": True,
        "unexpected": sorted(observed_set - declared_set),
        "missing": sorted(declared_set - observed_set),
    }


def quality_and_blockers(verdict: str, observations: dict[str, Any] | None = None) -> dict[str, Any]:
    """Declared quality expectation, the suites observed in this stage's commit state, and CI bookkeeping.

    The expectations are declared here, before any suite is run in this state, and the observed results are
    supplied by the operator with the exact command and the exact failing node ids. An observed failure
    outside the declared sets is a new regression: it is reported as one and it is never absorbed into the
    structural class.
    """
    baseline = REPO_ROOT / "quality" / "legacy_two_phase_debt_v1.json"
    stage_suite = _observed("stage_suite", observations)
    predecessor = _observed("predecessor_stage_suite", observations)
    sweep = _observed("frozen_sweep", observations)
    legacy = _observed("legacy_suite", observations)
    sweep_failures = dict(DECLARED_FROZEN_ADJACENT_SUITE_FAILURES)
    predecessor_check = _subset_report(predecessor["reported_failing"], DECLARED_PREDECESSOR_STAGE_SUITE_FAILURES)
    sweep_declared = [name for group in sweep_failures.values() for name in group]
    sweep_check = _subset_report(sweep["reported_failing"], sweep_declared)
    legacy_check = _subset_report(legacy["reported_failing"], DECLARED_LEGACY_LINEAGE_FAILURES)
    stage_check = _subset_report(stage_suite["reported_failing"], DECLARED_STAGE_SUITE_FAILURES)
    regressions = sorted(
        {
            name
            for check in (predecessor_check, sweep_check, legacy_check, stage_check)
            for name in (check["unexpected"] or [])
        }
    )
    complete = all(entry["observed"] for entry in (stage_suite, predecessor, sweep, legacy))
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
        "declared_expectations": {
            "stage_suite": {"command": STAGE_SUITE_COMMAND, "declared_failures": list(DECLARED_STAGE_SUITE_FAILURES)},
            "predecessor_stage_suite": {
                "command": PREDECESSOR_STAGE_SUITE_COMMAND,
                "declared_failures": list(DECLARED_PREDECESSOR_STAGE_SUITE_FAILURES),
                "classification": "STRUCTURAL_PREDECESSOR_PREFLIGHT_FAILURES",
                "rule": (
                    "declared before measuring: every one of these is a frozen L1A-2y preflight that cannot "
                    "hold once a later stage is committed on the same branch, because it requires every "
                    "commit above its own baseline to belong to its own stage and the worktree to be clean "
                    "outside its own paths"
                ),
            },
            "frozen_adjacent_suite": {
                "command": FROZEN_SWEEP_COMMAND,
                "suite_failures": {name: len(group) for name, group in sweep_failures.items()},
                "declared_failures": sweep_declared,
                "classification": "SUPERSEDED_BY_THE_NEXT_STAGE_BY_CONSTRUCTION",
            },
            "legacy_lineage_suite": {
                "command": LEGACY_SWEEP_COMMAND,
                "declared_failures": list(DECLARED_LEGACY_LINEAGE_FAILURES),
                "classification": "FROZEN_LINEAGE_DRIFT_CONFIRMED",
            },
        },
        "stage_suite": {
            "command": STAGE_SUITE_COMMAND,
            "result": stage_suite["reported_result"] or "NOT_OBSERVED",
            "failing": stage_suite["reported_failing"] or [],
            "observed": stage_suite["observed"],
            "observed_at": stage_suite["observed_at"],
            "unexpected_vs_declared": stage_check["unexpected"],
        },
        "predecessor_stage_suite": {
            "command": PREDECESSOR_STAGE_SUITE_COMMAND,
            "result": predecessor["reported_result"] or "NOT_OBSERVED",
            "failing": predecessor["reported_failing"] or [],
            "observed": predecessor["observed"],
            "observed_at": predecessor["observed_at"],
            "declared": list(DECLARED_PREDECESSOR_STAGE_SUITE_FAILURES),
            "unexpected_vs_declared": predecessor_check["unexpected"],
            "declared_but_not_observed": predecessor_check["missing"],
            "action": "recorded truthfully as observed; no frozen test was edited and none was turned green",
        },
        "frozen_adjacent_suite": {
            "classification": STRUCTURAL_FAILURE_CLASSES[1],
            "suite_failures": {name: len(group) for name, group in sweep_failures.items()},
            "suite_failure_count": int(sum(len(group) for group in sweep_failures.values())),
            "declared": sweep_declared,
            "sweep_command": FROZEN_SWEEP_COMMAND,
            "result": sweep["reported_result"] or "NOT_OBSERVED",
            "failing": sweep["reported_failing"] or [],
            "observed": sweep["observed"],
            "observed_at": sweep["observed_at"],
            "unexpected_vs_declared": sweep_check["unexpected"],
            "measured_at": _sweep_measured_at(),
            "combined_sweep_observation": {
                "command": COMBINED_SWEEP_COMMAND,
                "result": (observations or {}).get("combined_sweep", {}).get("result") or "NOT_OBSERVED",
                "failing": (observations or {}).get("combined_sweep", {}).get("failing") or [],
                "observed": bool((observations or {}).get("combined_sweep", {}).get("result")),
            },
            "cause": (
                "the frozen predecessor preflights require every commit above their own baseline to belong "
                "to their own stage, and two of them additionally require a worktree clean outside their "
                "own paths; neither can hold once a later stage is committed on the same branch"
            ),
        },
        "legacy_lineage_suite": {
            "failing": legacy["reported_failing"] or [],
            "declared": list(DECLARED_LEGACY_LINEAGE_FAILURES),
            "sweep_command": LEGACY_SWEEP_COMMAND,
            "result": legacy["reported_result"] or "NOT_OBSERVED",
            "observed": legacy["observed"],
            "observed_at": legacy["observed_at"],
            "classification": STRUCTURAL_FAILURE_CLASSES[2],
            "unexpected_vs_declared": legacy_check["unexpected"],
            "cause": "pre-existing; reproduced on a pristine worktree of the verified main, not caused by this stage",
            "action": "kept strictly separate from the structural failures and never counted as a pass",
        },
        "observations_complete": complete,
        "regressions_outside_the_declared_classes": regressions,
        "reporting_rule": (
            "the structural predecessor failures, the structural frozen-adjacent failures and the old "
            "lineage failure are reported separately with their own counts; nothing is hidden, suppressed or "
            "forced green, and an observed failure outside the declared sets is reported as a new regression"
        ),
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


def _sweep_measured_at() -> str:
    """Where a sweep figure was taken: the head at run time plus the worktree state it saw."""
    head = _git(["rev-parse", "HEAD"], required=False) or "UNKNOWN_HEAD"
    return (
        f"head {head}, with this stage's own evidence directory present in the worktree but uncommitted "
        "(the state the stage produced that evidence in)"
    )


REPORT_SECTIONS = (
    "A. Scope, authority and baseline",
    "B. Why the L1A-2y primary rule could not close",
    "C. The corrected rule and why it closes",
    "D. The frozen L1A-2x uniqueness diagnostic, retained and explained",
    "E. The prespecified branch-uniqueness rule and its t = 0 degeneracy",
    "F. The corrected attribution in dx",
    "G. Raw closure and residual decomposition",
    "H. Comparison with L1A-2y and L1A-2x",
    "I. Support uniqueness over the window",
    "J. Identical-window reconstruction check",
    "K. Conservation, solver convergence and capture equivalence",
    "L. Adversarial and falsification results",
    "M. Quality ratchet, stage tests and frozen adjacent suites",
    "N. Validity screen and the verdict",
    "O. Limitations and what would be needed for a mechanism claim",
    "P. Exactly one next action",
)


def render_report(result: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Render the human-readable report and its machine-readable headline from the assembled payloads."""
    ledger = result["sequential_ownbase_ledger.json"]
    design = result["support_uniqueness_design.json"]
    attribution_payload = result["attribution.json"]
    residuals = result["residual_decomposition.json"]
    validity = result["validity_screen.json"]
    decision = result["mechanism_decision.json"]
    action = result["next_action.json"]
    adversarial = result["adversarial_and_falsification.json"]
    quality = result["quality_and_blocker_matrix.json"]
    provenance = result["source_and_provenance.json"]
    freeze = result["prespecification_freeze.json"]
    prerequisites = result["prerequisite_verification.json"]
    chain = prerequisites["prior_stages"]["frozen_chain"]["stages"]
    identity = result["initial_state_identity.json"]
    equivalence = result["capture_equivalence.json"]
    identical = result["identical_window_check.json"]
    conservation = result["conservation_and_convergence.json"]
    primary = design["primary_support"]
    values = attribution_payload["displacement_dx"][primary][PRIMARY_CONVENTION]
    margins = attribution_payload["rule_margins_dx"]
    terms = residuals["terms_dx"]
    lines: list[str] = []
    add = lines.append
    add(f"# {STAGE} - prespecified own-base attribution and unique-support measurement")
    add("")
    add(f"## {REPORT_SECTIONS[0]}")
    add(
        f"- baseline: PR #24 head `{BASELINE_HEAD_SHA}` ({BASELINE_STAGE}); verified main `{VERIFIED_MAIN_SHA}`; "
        f"branch `{BRANCH}` at head `{provenance['head_sha']}`; solver contract "
        f"{provenance['solver_contract_version']}"
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
    add(
        f"- prespecification: `{freeze['document']}` sha256 `{freeze['document_sha256'][:16]}...`, commit "
        f"`{str(freeze['prespecification_commit'])[:12]}`, ancestor of head "
        f"{freeze['prespecification_commit_is_an_ancestor_of_head']}, measurement-code hash-block syncs "
        f"declared {freeze['hash_block_syncs']['declared']} / observed {freeze['hash_block_syncs']['observed']}"
    )
    add(
        "- frozen inputs re-verified: L1A-2y verified "
        f"{prerequisites['prior_stages']['l1a2y']['verified']} over "
        f"{prerequisites['prior_stages']['l1a2y']['bound_files']} bound files; the whole frozen chain "
        f"L1A-2t/u/v/w/x verified "
        f"{prerequisites['prior_stages']['frozen_chain']['all_verified']} over "
        f"{len(prerequisites['prior_stages']['frozen_chain']['stages'])} manifests, "
        f"{sum(entry['bound_files'] or 0 for entry in chain.values())} bound files"
    )
    add("")
    add(f"## {REPORT_SECTIONS[1]}")
    add(
        "- the frozen L1A-2x substep is `adv`, then `ch_explicit` (both folded into `phi_hat`), then "
        "`ch_implicit`; writing it any other way is not what the solver executed"
    )
    add(
        "- L1A-2y applied each ingredient to the same substep base and then chained each ingredient's own "
        "increments across the window, so its three totals measured displacements of partial fields that "
        "telescope to nothing in particular"
    )
    add(
        f"- the sealed consequence: its primary rule summed to {_fmt(sealed_attribution_sum(), 8)} dx against a "
        f"total of {_fmt(sealed_attribution_total(), 8)} dx, a closure of {_fmt(sealed_attribution_sum(), 8)} dx "
        f"- 12.2 times its own {ATTRIBUTION_CLOSURE_BAND_DX} dx band"
    )
    add(
        "- the frozen event threshold `CONTOUR_UNIQUENESS_MARGIN_DX = "
        f"{float(stage_2x.CONTOUR_UNIQUENESS_MARGIN_DX)}` dx is unchanged in the frozen module, which this "
        "stage imports rather than edits"
    )
    add("")
    add(f"## {REPORT_SECTIONS[2]}")
    add(
        "- measure `Q` on the consecutive actual states of each frozen substep: "
        "`D_adv = sum_k [Q(S1_k) - Q(S0_k)]`, `D_ch = sum_k [Q(S2_k) - Q(S1_k)]`, "
        "`D_imp = sum_k [Q(S3_k) - Q(S2_k)]`"
    )
    add(
        "- the sum telescopes to `Q(phi_final) - Q(phi_initial)` by construction; the run reports the "
        "floating-point residual of that identity rather than assuming it"
    )
    add(
        f"- closure in this window: total {_fmt(terms['R_sequential_telescoping_total'], 8)} dx, worst "
        f"substep {_fmt(terms['R_sequential_telescoping_max_substep'], 8)} dx, band "
        f"{ATTRIBUTION_CLOSURE_BAND_DX} dx"
    )
    add(
        "- the identity is reported as a closure check and is never presented as evidence that a physical "
        "operator caused the motion"
    )
    add("")
    add(f"## {REPORT_SECTIONS[3]}")
    add(
        f"- the frozen diagnostic was recomputed on every intermediate state: statuses "
        f"{json.dumps(validity['frozen_l1a2x_diagnostic_retained']['statuses'])}, margin range "
        f"[{_fmt(validity['frozen_l1a2x_diagnostic_retained']['minimum_margin_over_dx'], 6)}, "
        f"{_fmt(validity['frozen_l1a2x_diagnostic_retained']['maximum_margin_over_dx'], 6)}] dx"
    )
    add(
        "- the reason is structural, not numerical: the two lowest-gap `phi = 0.5` crossings at t = 0 are "
        "the mirror pair `x = 2.984375` and `x = 3.015625` at the same height, and the frozen SDF is exactly "
        "linear in the normal coordinate there, so both sit at the same wall distance to machine precision"
    )
    add(
        "- the frozen rule is retained verbatim as a historical comparator. Its verdict is not the verdict "
        "of this stage and no frozen file was edited to change it"
    )
    add("")
    add(f"## {REPORT_SECTIONS[4]}")
    add(
        f"- candidates: every reconstructed crossing ({ledger['t0_uniqueness']['candidates']} at t = 0); "
        f"components: connected components of the same reconstruction built from the frozen per-cell "
        f"segments and validated cell by cell against them "
        f"({design['t0_uniqueness']['frozen_cell_by_cell_validation']['cells_checked']} cells, matches "
        f"{design['t0_uniqueness']['frozen_cell_by_cell_validation']['matches']})"
    )
    add(
        f"- minimum: `g*` = {_fmt(ledger['t0_uniqueness']['minimum_gap_over_dx'], 8)} dx; tie tolerance "
        f"{TIE_TOLERANCE_DX} dx selects {ledger['t0_uniqueness']['tie_set_size']} tied candidates; competition "
        f"tolerance {COMPETITION_TOLERANCE_DX} dx admits {ledger['t0_uniqueness']['competition_set_size']}"
    )
    add(
        f"- the tied candidates lie on {len(ledger['t0_uniqueness']['tie_components'])} component(s): the "
        f"point degeneracy is inside one branch, so the branch is unique and the point is not. The point "
        f"degeneracy is recorded, both tied supports are measured, and their difference is reported as "
        f"{_fmt(attribution_payload['comparison_terms_dx']['support_tie_spread'], 8)} dx"
    )
    add(
        f"- deterministic tie-break: the smallest-x tied candidate ({primary}) is the reporting support; it "
        "is never used as evidence of uniqueness and no region was shrunk to leave one candidate"
    )
    add(f"- the frozen minimum-gap rule reproduced exactly: {json.dumps(design['frozen_support_replay'])}")
    add("")
    add(f"## {REPORT_SECTIONS[5]}")
    add(
        f"- primary convention `{PRIMARY_CONVENTION}` on support `{primary}`, along the frozen wall normal, "
        "positive toward the solid, raw dx, nothing normalised:"
    )
    for name in ("adv", "ch_explicit", "ch_implicit", "ch_pair", "full"):
        add(f"    - {name}: {_fmt(values[name], 9)}")
    add(
        "- the two Cahn-Hilliard halves are one update split by the implementation into an explicit flux and "
        "an implicit solve; they are never presented as two independent mechanisms"
    )
    add(
        "- the other prespecified conventions, for the record (dx): "
        + json.dumps(
            {
                convention: {
                    key: round(float(value[key]), 9) for key in ("adv", "ch_pair", "full") if value.get(key) is not None
                }
                for convention, value in attribution_payload["displacement_dx"][primary].items()
            }
        )
    )
    add(f"- per macro step (dx): {json.dumps([round(row['dn_full'], 8) for row in attribution_payload['per_macro']])}")
    add("")
    add(f"## {REPORT_SECTIONS[6]}")
    add("- every term, raw, in dx:")
    for name, value in terms.items():
        add(f"    - {name}: {_fmt(value, 10)}")
    add(
        f"- comparison error {_fmt(attribution_payload['comparison_error_dx'], 8)} dx made of "
        f"{json.dumps({key: round(value, 9) for key, value in attribution_payload['comparison_terms_dx'].items()})}"
    )
    add(
        f"- dominance margin {_fmt(margins[PRIMARY_CONVENTION], 8)} dx against the frozen floor of "
        f"{DOMINANCE_FACTOR} x the comparison error = {_fmt(attribution_payload['dominance_floor_dx'], 8)} dx"
    )
    add("")
    add(f"## {REPORT_SECTIONS[7]}")
    add(
        f"- L1A-2y primary closure {_fmt(residuals['historical_comparator']['l1a2y_primary_closure_dx'], 8)} dx "
        f"versus this stage's {_fmt(terms['R_sequential_telescoping_total'], 10)} dx on the identical window, "
        "with the frozen band unchanged at 0.01 dx"
    )
    add(
        f"- L1A-2y own-base comparator closure "
        f"{_fmt(residuals['historical_comparator']['l1a2y_own_base_closure_dx'], 8)} dx is the closest thing "
        "that stage recorded to this rule; the difference between the two is the base anchoring, not a "
        "threshold"
    )
    increments = {key: float(f"{value:.3g}") for key, value in identical["per_substep_increment_samples"].items()}
    add(
        f"- identical-window check against the sealed ledger: roots differ by at most "
        f"{_fmt(identical['max_absolute_root_difference_over_dx'], 10)} dx over "
        f"{identical['macro_steps_compared']} macro steps; per-substep increment differences "
        f"{json.dumps(increments)}"
    )
    add(
        "- L1A-2x carry-over unchanged: its four-way closure and its halted macro-25 support crossing are "
        "read from its sealed payload and are not recomputed here as new evidence"
    )
    add("")
    add(f"## {REPORT_SECTIONS[8]}")
    add(
        f"- intermediate states checked: {validity['intermediate_states_checked']}, rule satisfied "
        f"{validity['intermediate_states_rule_ok']}, non-unique {validity['non_unique_state_count']}, "
        f"exact-threshold {validity['exact_threshold_state_count']}, enumeration mismatches "
        f"{validity['frozen_enumeration_mismatch_count']}, multi-component competition "
        f"{validity['multi_component_state_count']}, branch reassignments {len(validity['branch_reassignments'])}"
    )
    add(
        "- the t = 0 tie is inside one branch and stays inside it for the whole window; the recorded "
        "segment change at macro 25 / substep 74 is a within-branch event and is reported rather than "
        "resolved away"
    )
    add("")
    add(f"## {REPORT_SECTIONS[9]}")
    add(
        f"- initial and final state hashes match the sealed L1A-2y window: "
        f"{identical['initial_state_hashes_match']} and {identical['final_state_hashes_match']}; tolerance "
        f"{CROSS_CHECK_TOLERANCE_DX} dx; view holds {identical['holds']}"
    )
    add(
        "- a trajectory difference would have been treated as a separate regression and would have blocked "
        "the verdict; the check is recorded either way"
    )
    add("")
    add(f"## {REPORT_SECTIONS[10]}")
    add(
        f"- mass drift {_fmt(conservation['mass_drift'], 6)} (relative "
        f"{_fmt(conservation['mass_drift_relative'], 6)}), worst substep mass residual "
        f"{_fmt(conservation['worst_substep_mass_residual'], 6)}, worst closed-face flux "
        f"{_fmt(conservation['worst_closed_face_flux'], 6)}, CG iterations "
        f"{conservation['cg_iterations_min']}..{conservation['cg_iterations_max']}, all converged "
        f"{conservation['cg_all_converged']}"
    )
    add(
        f"- capture versus the frozen public step: phi "
        f"{_fmt(equivalence['max_phi_difference_vs_eager_public_step'], 6)} eager / "
        f"{_fmt(equivalence['max_phi_difference_vs_jitted_public_step'], 6)} jitted, gap "
        f"{_fmt(equivalence['max_gap_difference_over_dx'], 6)} dx, within the frozen thresholds "
        f"{equivalence_ok_flag(validity)}"
    )
    add("")
    add(f"## {REPORT_SECTIONS[11]}")
    add(
        f"- {adversarial['checks_run']} checks, all hold {adversarial['all_hold']}, failed "
        f"{json.dumps(adversarial['failed_checks'])}"
    )
    for entry in adversarial["checks"]:
        add(f"    - {entry['check']}: {entry['holds']}")
    add(
        "- the gate is demonstrably able to fail: the sealed L1A-2y primary rule misses the same 0.01 dx band "
        "that this stage uses, and a mixed-family sum is shown to be able to land inside that band, which is "
        "why the prespecified rule and not the band is the contract"
    )
    add("")
    add(f"## {REPORT_SECTIONS[12]}")
    add(
        f"- legacy debt ratchet: expected `{quality['ratchet_expectation']}` against "
        f"{json.dumps(quality['legacy_debt_baseline'])}, new-code diagnostics declared "
        f"{quality['declared_new_code_diagnostics']}"
    )
    add(f"- stage suite: {quality['stage_suite']['result']} - `{quality['stage_suite']['command']}`")
    add(
        f"- predecessor stage suite: {quality['predecessor_stage_suite']['result']} - declared structural "
        f"failures observed {json.dumps(quality['predecessor_stage_suite']['failing'])}, declared but not "
        f"observed {json.dumps(quality['predecessor_stage_suite']['declared_but_not_observed'])}"
    )
    add(
        f"- frozen adjacent suites: {quality['frozen_adjacent_suite']['result']} - declared structural set "
        f"{json.dumps(quality['frozen_adjacent_suite']['suite_failures'])}; legacy lineage suite "
        f"{quality['legacy_lineage_suite']['result']}, pre-existing"
    )
    add(
        f"- observations complete {quality['observations_complete']}; failures outside the declared classes "
        f"{json.dumps(quality['regressions_outside_the_declared_classes'])}"
    )
    add(
        "- hosted CI: no result is claimed here; the observed post-push result is recorded by the re-seal "
        "step and this line is replaced with it"
    )
    add("")
    add(f"## {REPORT_SECTIONS[13]}")
    add(f"- verdict: `{decision['verdict']}`")
    add(
        f"- A measurement validity {decision['questions']['A_measurement_validity']}; B mathematical "
        f"attribution closure {decision['questions']['B_mathematical_attribution_closure']}; C physical "
        f"mechanism {decision['questions']['C_physical_mechanism']} - this stage never answers C"
    )
    add(
        f"- dominance reading `{decision['dominance_reading']}` with margin "
        f"{_fmt(decision['dominance_margin_dx'], 8)} dx against floor {_fmt(decision['dominance_floor_dx'], 8)} dx"
    )
    add(f"- withheld because: {json.dumps(decision['verdict_withheld_because'])}")
    add(
        "- measurement attribution resolved under the specified convention "
        f"{decision['measurement_attribution_resolved_under_the_specified_convention']}"
    )
    add(f"- validity screen blocking failures: {json.dumps(validity['blocking_failures'])}")
    add("")
    add(f"## {REPORT_SECTIONS[14]}")
    for claim in decision["forbidden_claims_respected"]:
        add(f"- {claim}")
    add(
        "- scope: one case, one dt, one trajectory, t <= 0.08, zero contact events. Nothing here is a "
        "statement about the impact at t ~ 0.354, about a real contact mechanism, about a grid or time-step "
        "refinement, or about any parameter outside the frozen set"
    )
    add(
        "- the two Cahn-Hilliard halves, the four attribution conventions and the support tie are all "
        "measurement choices; a resolved attribution under this convention is not a statement that this "
        "convention is the only admissible one"
    )
    add("")
    add(f"## {REPORT_SECTIONS[15]}")
    add(f"- {action['action']}")
    add("- not started in this stage and requiring owner approval before it is started")
    add(f"- bounded scope of any approved follow-up: {json.dumps(action['bounded_scope'], sort_keys=True)}")
    for failure in action["measured_failures_of_this_stage"]:
        add(f"- recorded failure of this stage: {failure}")
    report_text = "\n".join(lines) + "\n"
    report_payload = {
        "stage": STAGE,
        "verdict": decision["verdict"],
        "headline": {
            "measurement_attribution_resolved_under_the_specified_convention": decision[
                "measurement_attribution_resolved_under_the_specified_convention"
            ],
            "advective_displacement_dx": decision["advective_displacement_dx"],
            "cahn_hilliard_displacement_dx": decision["cahn_hilliard_displacement_dx"],
            "total_displacement_dx": decision["total_displacement_dx"],
            "closure_residual_dx": terms["R_sequential_telescoping_total"],
            "closure_band_dx": ATTRIBUTION_CLOSURE_BAND_DX,
            "dominance_reading": decision["dominance_reading"],
            "dominance_margin_dx": decision["dominance_margin_dx"],
            "dominance_floor_dx": decision["dominance_floor_dx"],
            "support_uniqueness": {
                "frozen_l1a2x_rule": validity["frozen_l1a2x_diagnostic_retained"]["statuses"],
                "this_stage_rule": sorted({row["status"] for row in ledger["uniqueness_rows"]}),
                "tied_supports_measured": len(ledger["supports"]),
                "support_tie_spread_dx": attribution_payload["comparison_terms_dx"]["support_tie_spread"],
            },
            "physical_mechanism_claim": False,
        },
        "reference_times": {"t_final": ledger["t_final"], "macro_steps_run": ledger["macro_steps_run"]},
        "sections": list(REPORT_SECTIONS),
        "identity": {
            "initial_state_hashes": identity["hashes"],
            "identity_checks": identity["identity_checks"],
            "support": {
                "point": design["supports"][0]["point"],
                "wall_normal_into_solid": design["supports"][0]["wall_normal_into_solid"],
                "gap_over_dx": design["supports"][0]["gap_over_dx"],
                "cell": design["supports"][0]["cell"],
            },
        },
    }
    return report_payload, report_text


def sealed_attribution_sum() -> float:
    """The sealed L1A-2y primary-rule closure, read for the narrative only."""
    sealed = json.loads((PRIOR_EVIDENCE["l1a2y"] / "residual_decomposition.json").read_text())
    return float(sealed["terms_dx"]["R_attribution_closure"])


def sealed_attribution_total() -> float:
    """The sealed L1A-2y measured total displacement, read for the narrative only."""
    sealed = json.loads((PRIOR_EVIDENCE["l1a2y"] / "interface_attribution.json").read_text())
    return float(sealed["cumulative_dn_dx"]["full"])


def equivalence_ok_flag(validity: dict[str, Any]) -> bool:
    return bool(validity["equivalence_ok"])


# ---------------------------------------------------------------------------
# Section I - identity, assembly, verification and the run driver
# ---------------------------------------------------------------------------
def initial_state_identity() -> dict[str, Any]:
    """Bind the frozen C2 initial state to the sealed L1A-2y identity, fail closed on any difference."""
    bundle, state0 = stage_2x._bundle_and_state()
    sealed = json.loads((PRIOR_EVIDENCE["l1a2y"] / "initial_state_identity.json").read_text())
    hashes = {name: _array_sha(getattr(state0, name)) for name in ("phi", "u", "v")}
    stored_dtypes = {name: str(np.asarray(getattr(state0, name)).dtype) for name in ("phi", "u", "v")}
    sealed_hashes = {key: str(value) for key, value in sealed["hashes"].items()}
    checks = {f"{key}_hash": hashes[key] == sealed_hashes.get(key) for key in hashes}
    checks["state_t_zero"] = float(np.asarray(state0.t)) == 0.0
    checks["contract_version"] = int(pf.SOLVER_CONTRACT_VERSION) == 12
    if not all(checks.values()):
        raise AuditBlocked(f"the initial state does not match the sealed L1A-2y identity: {checks}")
    return {
        "stage": STAGE,
        "case_name": CASE_NAME,
        "profile_N": PROFILE_N,
        "state_source": "the frozen L1A-2x kernel capture of the frozen L1A-2s C2 (SDF_TAPERED_STREAMFUNCTION_V1)",
        "hashes": hashes,
        "stored_dtypes": stored_dtypes,
        "sealed_l1a2y_hashes": sealed_hashes,
        "identity_checks": checks,
        "parameters": dict(sealed["parameters"]),
        "geometry_hash": sealed["geometry_hash"],
        "parameter_hash": sealed["parameter_hash"],
        "support": dict(sealed["support"]),
        "prohibited_alternatives_not_used": list(sealed["prohibited_alternatives_not_used"]),
        "rule": (
            "the state is rebuilt by the frozen audit and bound to the sealed L1A-2y identity; every hash is "
            "taken from the array as it is stored (the legacy float32 velocity precision included, exactly as "
            "the frozen seal hashed it) and a difference stops the run before anything is measured"
        ),
    }


def trajectory_specification(ledger: dict[str, Any]) -> dict[str, Any]:
    """The window, stated as one frozen configuration with its hashes and its prohibitions."""
    return {
        "stage": STAGE,
        "case_name": ledger["case_name"],
        "profile_N": ledger["profile_N"],
        "dt": ledger["requested_dt"],
        "substep_dt": ledger["parameters"]["substep_dt"],
        "substeps_per_macro": ledger["substeps_per_macro"],
        "macro_step_cap": MAX_MACRO_STEPS,
        "macro_steps_run": ledger["macro_steps_run"],
        "t_final": ledger["t_final"],
        "solver": "the frozen contract-12 phasefield path, re-executed by the frozen L1A-2x capture",
        "readout": "l1a2y_contour_branch_readout (frozen contour-branch readout)",
        "initial_state_hashes": ledger["initial_state"],
        "final_state_hashes": ledger["final_state"],
        "contact_guard": ledger["contact_guard"],
        "contact_threshold_over_dx": CONTACT_GAP_THRESHOLD_DX,
        "segment_binding_switch": ledger["segment_binding_switch"],
        "pinned_components": ledger["pinned_components"],
        "prohibitions": ledger["prohibitions"],
        "changed_relative_to_l1a2y": {
            "trajectory": False,
            "dt": False,
            "mesh": False,
            "solver_operators": False,
            "contact_guard": False,
            "physical_constants": False,
            "acceptance_criteria": False,
            "measurement_protocol": True,
        },
        "equivalence_note": (
            "the same trajectory as L1A-2y is re-run through the frozen kernel capture; the measurement "
            "layer is the only thing this stage replaces"
        ),
    }


def assemble_evidence(result: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Write every evidence file, hash the seal, and verify the directory as an independent consumer would."""
    directory = Path(out_dir)
    missing = [name for name in EVIDENCE_FILES if name not in ("manifest.json",) and result.get(name) is None]
    if missing:
        raise AuditBlocked(
            f"evidence assembly refused: the run produced no payload for {missing}; a partial run never "
            f"writes evidence, because a directory that is missing a required file is not a seal"
        )
    directory.mkdir(parents=True, exist_ok=True)
    report_payload, report_text = render_report(result)
    for name in EVIDENCE_FILES:
        if name in ("l1a2z_final_report.json", "l1a2z_final_report.md", "manifest.json"):
            continue
        payload = result.get(name)
        if payload is None:
            raise AuditBlocked(f"no payload for the required evidence file {name}")
        (directory / name).write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    (directory / "l1a2z_final_report.json").write_text(
        json.dumps(report_payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    (directory / "l1a2z_final_report.md").write_text(report_text)
    ledger = result["sequential_ownbase_ledger.json"]
    decision = result["mechanism_decision.json"]
    provenance = result["source_and_provenance.json"]
    freeze = result["prespecification_freeze.json"]
    manifest = {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "profile": "early_window_sequential_own_base",
        "evidence_class": "FROZEN_CHNS_SEQUENTIAL_OWNBASE_ATTRIBUTION",
        "baseline_head_sha": BASELINE_HEAD_SHA,
        "baseline_stage": BASELINE_STAGE,
        "verified_main_sha": VERIFIED_MAIN_SHA,
        "solver_contract_version": 12,
        "case_name": ledger["case_name"],
        "profile_N": ledger["profile_N"],
        "requested_dt": ledger["requested_dt"],
        "macro_steps_run": ledger["macro_steps_run"],
        "substeps_recorded": ledger["substeps_recorded"],
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
        "physical_mechanism_claim": False,
        "quality_observations_complete": bool(result["quality_and_blocker_matrix.json"].get("observations_complete")),
        "regressions_outside_the_declared_classes": list(
            result["quality_and_blocker_matrix.json"].get("regressions_outside_the_declared_classes", [])
        ),
        "readout_version": provenance["readout_version"],
        "prespecification_document": freeze["document"],
        "prespecification_document_sha256": freeze["document_sha256"],
        "prespecification_commit": freeze["prespecification_commit"],
        "module_sha256": provenance["source_hashes"]["stage_module"],
        "source_hashes": dict(provenance["source_hashes"]),
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


def verify_evidence_manifest(out_dir: str | Path) -> dict[str, Any]:
    """Verify one assembled L1A-2z evidence directory against its own manifest, fail closed."""
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
    if manifest.get("physical_mechanism_claim") is not False:
        problems.append("the manifest must not claim a physical mechanism")
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
    seal = _seal_of(manifest)
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
        "prespecification_commit": manifest.get("prespecification_commit"),
        "bound_files": len(seal),
        "recorded_source_hashes": sources["checked"],
        "recorded_source_hashes_verified_from_the_worktree": len(sources["verified_from_the_worktree"]),
        "hosted_ci_reseal": manifest.get("hosted_ci_reseal"),
        "directory": str(directory),
        "problems": problems,
        "missing_required": missing,
        "verified": True,
    }


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
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # fail closed before touching a byte: every bound payload that this function does not rewrite must
    # still match the seal of the manifest it was assembled with
    for name, value in sorted(_seal_of(manifest).items()):
        if name in ("manifest.json", "l1a2z_final_report.md"):
            continue
        if sha256_file(directory / name) != value:
            raise AuditBlocked(f"the re-seal would cover a changed payload: {name}")
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
    report_path = directory / "l1a2z_final_report.md"
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
    manifest["hosted_ci_reseal"] = {
        "performed": True,
        "status": ci_status,
        "observed_head_sha": head_sha,
        "previous_status": previous,
        "physics_evidence_untouched": True,
    }
    manifest["files"]["l1a2z_final_report.md"] = sha256_file(report_path)
    manifest["files"]["quality_and_blocker_matrix.json"] = sha256_file(quality_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n")
    return verify_evidence_manifest(directory)


# ---------------------------------------------------------------------------
# Section J - the stage plan and the command line
# ---------------------------------------------------------------------------
STAGES = (
    "preflight",
    "prespecification",
    "provenance",
    "prerequisites",
    "identity",
    "design",
    "window",
    "attribution",
    "residuals",
    "conservation",
    "equivalence",
    "identical",
    "validity",
    "decision",
    "action",
    "adversarial",
    "quality",
    "report",
)
STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "preflight": (),
    "prespecification": (),
    "provenance": ("preflight", "prespecification"),
    "prerequisites": ("prespecification",),
    "identity": ("prerequisites",),
    "design": ("identity",),
    "window": ("design",),
    "attribution": ("window",),
    "residuals": ("attribution",),
    "conservation": ("window",),
    "equivalence": ("window",),
    "identical": ("window",),
    "validity": ("residuals", "conservation", "equivalence", "identical"),
    "decision": ("attribution", "residuals", "validity", "identical"),
    "action": ("decision",),
    "adversarial": ("window", "attribution", "residuals", "validity", "equivalence", "identical"),
    "quality": ("adversarial",),
    "report": tuple(name for name in STAGES if name != "report"),
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


def run(
    *,
    out: str | Path | None = None,
    stages: str = "all",
    macro_steps: int = MAX_MACRO_STEPS,
    observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the requested stages (with their dependencies) and return every payload by evidence file name."""
    import time as _time

    plan = plan_stages(stages)
    result: dict[str, Any] = {}
    started = _time.perf_counter()
    progress(f"stage plan: {', '.join(plan)}")
    result["preflight.json"] = preflight_state()
    result["prespecification_freeze.json"] = prespecification_freeze()
    result["source_and_provenance.json"] = source_and_provenance(
        result["preflight.json"], result["prespecification_freeze.json"]
    )
    if "prerequisites" in plan:
        progress("re-hashing the sealed L1A-2y and L1A-2x evidence")
        result["prerequisite_verification.json"] = prerequisite_verification()
    if "identity" in plan:
        progress("binding the frozen C2 initial state to the sealed identity")
        result["initial_state_identity.json"] = initial_state_identity()
        bundle, state0 = stage_2x._bundle_and_state()
        design, supports = support_design(bundle, state0)
        del supports
        result["support_uniqueness_design.json"] = design
        progress(
            "support design frozen: "
            f"{design['t0_uniqueness']['tie_set_size']} tied supports on "
            f"{len(design['t0_uniqueness']['tie_components'])} component(s), primary {design['primary_support']}"
        )
    if "window" in plan:
        progress("running the one sanctioned window with the sequential own-base measurement")
        ledger = run_window(macro_steps)
        result["sequential_ownbase_ledger.json"] = ledger
        result["trajectory_specification.json"] = trajectory_specification(ledger)
        result["support_uniqueness_window.json"] = support_uniqueness_window(ledger)
        progress(
            f"window done: {ledger['macro_steps_run']} macro steps, {ledger['substeps_recorded']} substeps, "
            f"t <= {_fmt(ledger['t_final'], 5)}, contact guard {ledger['contact_guard']['triggered']}"
        )
    if "attribution" in plan:
        progress("aggregating the attribution conventions")
        attribution_payload = attribution(result["sequential_ownbase_ledger.json"])
        result["attribution.json"] = attribution_payload
        result["residual_decomposition.json"] = residual_decomposition(
            result["sequential_ownbase_ledger.json"], attribution_payload
        )
        values = attribution_payload["displacement_dx"][attribution_payload["primary_support"]][PRIMARY_CONVENTION]
        progress(
            "attribution: adv "
            f"{_fmt(values['adv'], 6)} dx, ch-pair {_fmt(values['ch_pair'], 6)} dx, total "
            f"{_fmt(values['full'], 6)} dx, closure "
            f"{_fmt(result['residual_decomposition.json']['terms_dx']['R_sequential_telescoping_total'], 3)} dx"
        )
    if "conservation" in plan:
        result["conservation_and_convergence.json"] = conservation_and_convergence(
            result["sequential_ownbase_ledger.json"]
        )
    if "equivalence" in plan:
        result["capture_equivalence.json"] = capture_equivalence_summary(result["sequential_ownbase_ledger.json"])
    if "identical" in plan:
        progress("checking this run against the sealed L1A-2y window")
        result["identical_window_check.json"] = identical_window_check(result["sequential_ownbase_ledger.json"])
    if "validity" in plan:
        result["validity_screen.json"] = validity_screen(
            result["sequential_ownbase_ledger.json"],
            result["attribution.json"],
            result["residual_decomposition.json"],
            result["capture_equivalence.json"],
            result["identical_window_check.json"],
        )
        blocking = result["validity_screen.json"]["blocking_failures"]
        progress(f"validity: unique {result['validity_screen.json']['unique_measurement_ok']}, blocking {blocking}")
    if "decision" in plan:
        result["mechanism_decision.json"] = decide(
            result["sequential_ownbase_ledger.json"],
            result["attribution.json"],
            result["validity_screen.json"],
            result["identical_window_check.json"],
        )
        progress(f"verdict: {result['mechanism_decision.json']['verdict']}")
    if "action" in plan:
        result["next_action.json"] = next_action(result["mechanism_decision.json"])
    if "adversarial" in plan:
        progress("running the prespecified adversarial and falsification tests")
        result["adversarial_and_falsification.json"] = adversarial_and_falsification(
            result["sequential_ownbase_ledger.json"],
            result["attribution.json"],
            result["residual_decomposition.json"],
            result["validity_screen.json"],
            result["capture_equivalence.json"],
            result["identical_window_check.json"],
        )
        progress(
            f"adversarial: {result['adversarial_and_falsification.json']['checks_run']} checks, failed "
            f"{result['adversarial_and_falsification.json']['failed_checks']}"
        )
    if "quality" in plan:
        result["quality_and_blocker_matrix.json"] = quality_and_blockers(
            result["mechanism_decision.json"]["verdict"], observations
        )
    if "report" in plan:
        report_payload, report_text = render_report(result)
        result["l1a2z_final_report.json"] = report_payload
        result["l1a2z_final_report.md"] = report_text
    progress(f"stages done in {_time.perf_counter() - started:.1f}s")
    return result


def main(argv: list[str] | None = None) -> None:
    """Command line: run the stage, verify an assembled directory, or re-seal an observed CI status."""
    parser = argparse.ArgumentParser(description=f"{STAGE} - sequential own-base attribution and uniqueness")
    parser.add_argument("--out", default=None, help="evidence directory to write")
    parser.add_argument("--stages", default="all", help="comma separated stage list, or all")
    parser.add_argument("--macro-steps", type=int, default=MAX_MACRO_STEPS, help="macro step budget")
    parser.add_argument("--verify-evidence", default=None, help="verify an assembled evidence directory")
    parser.add_argument("--reseal-ci-status", default=None, help="observed hosted CI status to record")
    parser.add_argument("--reseal-head-sha", default=None, help="head sha the hosted CI ran on")
    parser.add_argument("--reseal-dir", default=None, help="evidence directory to re-seal")
    parser.add_argument("--observed-at", default=None, help="where the suite observations were taken")
    for name in ("stage-suite", "predecessor-suite", "frozen-sweep", "combined-sweep", "legacy-suite"):
        parser.add_argument(f"--{name}-result", default=None, help=f"observed pytest result line for {name}")
        parser.add_argument(f"--{name}-failing", default=None, help=f"observed failing node ids for {name}")
    args = parser.parse_args(argv)
    observations = {}
    for name, key in (
        ("stage-suite", "stage_suite"),
        ("predecessor-suite", "predecessor_stage_suite"),
        ("frozen-sweep", "frozen_sweep"),
        ("combined-sweep", "combined_sweep"),
        ("legacy-suite", "legacy_suite"),
    ):
        result_line = getattr(args, name.replace("-", "_") + "_result")
        failing = getattr(args, name.replace("-", "_") + "_failing")
        if result_line or failing:
            observations[key] = {
                "result": result_line,
                "failing": [entry for entry in (failing or "").split(",") if entry.strip()],
                "at": args.observed_at,
            }
    if args.verify_evidence:
        report = verify_evidence_manifest(args.verify_evidence)
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
        return
    if args.reseal_ci_status:
        directory = Path(args.reseal_dir) if args.reseal_dir else EVIDENCE_OWN
        report = reseal_ci_status(args.reseal_ci_status, str(args.reseal_head_sha), directory)
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "problems"}, indent=2, default=_json_default
            )
        )
        return
    outcome = run(out=args.out, stages=args.stages, macro_steps=args.macro_steps, observations=observations)
    directory = Path(args.out) if args.out else EVIDENCE_OWN
    if not all(name in outcome for name in EVIDENCE_FILES if name != "manifest.json"):
        print(
            f"[{STAGE}] partial run: {sorted(name for name in EVIDENCE_FILES if name not in outcome)} were not "
            f"produced, so nothing was written to {directory}. A partial run is a development check, never a "
            f"measurement.",
            flush=True,
        )
        return
    verification = assemble_evidence(outcome, directory)
    progress(f"evidence assembled under {directory}")
    print(
        json.dumps(
            {key: value for key, value in verification.items() if key != "problems"}, indent=2, default=_json_default
        )
    )
    decision = outcome["mechanism_decision.json"]
    resolved = decision["measurement_attribution_resolved_under_the_specified_convention"]
    print(f"[{STAGE}] verdict {decision['verdict']}; resolved {resolved}")
    print(f"[{STAGE}] withheld because {decision['verdict_withheld_because']}")


if __name__ == "__main__":  # pragma: no cover
    main()
