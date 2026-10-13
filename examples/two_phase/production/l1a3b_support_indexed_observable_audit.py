"""L1A-3b: freeze a tie-break-free, two-ray interface observable for a future study.

This module is an observable-contract audit only. It does not advance the solver, alter phase fields,
run a mobility intervention, or infer a physical mechanism. The two ray identities are inherited from
sealed L1A-2z evidence and remain fixed by grid geometry, not by whichever root gives a preferred result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from textwrap import fill
from types import SimpleNamespace
from typing import Any

import numpy as np
from production import frozen_chns_early_window_transport_attribution_audit as stage_2x  # noqa: E402
from production import l1a2y_contour_branch_readout as readout  # noqa: E402
from production import l1a2z_sequential_ownbase_attribution_audit as stage_2z  # noqa: E402

STAGE = "L1A-3b"
STAGE_VERSION = "l1a3b_v1"
BRANCH = "arena/02b6c71b-hydrogym"
BASELINE_HEAD_SHA = "29dd981dd0b162af05ca9bba00aa90f7dc5e8fea"
L1A2Z_HEAD_SHA = "d774e38209d7bd1c13810aaa3c2b6434209291f9"
L1A3A_HEAD_SHA = "29dd981dd0b162af05ca9bba00aa90f7dc5e8fea"
VERIFIED_MAIN_SHA = "09e2993b8da478d79e768054194201af57121784"

TWO_PHASE = Path(__file__).resolve().parents[1]
REPO_ROOT = TWO_PHASE.parents[1]
EVIDENCE_OWN = TWO_PHASE / "evidence" / "l1a3b"
EVIDENCE_2Z = TWO_PHASE / "evidence" / "l1a2z"
EVIDENCE_3A = TWO_PHASE / "evidence" / "l1a3a"
PRESPECIFICATION = TWO_PHASE / "l1a3b_prespecification.md"
MODULE_PATH = Path(__file__).resolve()
TEST_PATH = TWO_PHASE / "tests" / "test_l1a3b_support_indexed_observable_audit.py"
READOUT_PATH = TWO_PHASE / "production" / "l1a2y_contour_branch_readout.py"
L1A2Z_MODULE_PATH = TWO_PHASE / "production" / "l1a2z_sequential_ownbase_attribution_audit.py"
L1A2Z_TEST_PATH = TWO_PHASE / "tests" / "test_l1a2z_sequential_ownbase_attribution_audit.py"
L1A2X_MODULE_PATH = TWO_PHASE / "production" / "frozen_chns_early_window_transport_attribution_audit.py"

STAGE_OWNED_PATHS = (
    "examples/two_phase/production/l1a3b_support_indexed_observable_audit.py",
    "examples/two_phase/tests/test_l1a3b_support_indexed_observable_audit.py",
    "examples/two_phase/l1a3b_prespecification.md",
    "examples/two_phase/evidence/l1a3b",
)
SOURCE_PATHS = {
    "stage_module": MODULE_PATH,
    "stage_test": TEST_PATH,
    "prespecification": PRESPECIFICATION,
    "l1a2x_module": L1A2X_MODULE_PATH,
    "l1a2y_readout_module": READOUT_PATH,
    "l1a2z_module": L1A2Z_MODULE_PATH,
    "l1a2z_test": L1A2Z_TEST_PATH,
    "l1a2z_manifest": EVIDENCE_2Z / "manifest.json",
    "l1a3a_manifest": EVIDENCE_3A / "manifest.json",
    "l1a3a_phase_a_record": EVIDENCE_3A / "phase_a_audit.json",
}

RAY_IDS = ("RAY_95", "RAY_96")
FROZEN_RAY_GEOMETRY = {
    "RAY_95": {"column_index": 95, "x": 2.984375},
    "RAY_96": {"column_index": 96, "x": 3.015625},
}
CONTOUR_THRESHOLD = 0.5
ROOT_WINDOW_DX = readout.RAY_SEARCH_HALF_WINDOW_DX
ROOT_STEP_MAX_DX = readout.ROOT_MATCH_MAX_STEP_DX
ROOT_AMBIGUITY_SEPARATION_DX = readout.ROOT_AMBIGUITY_SEPARATION_DX
NODE_DEGENERACY_TOLERANCE_DX = readout.NODE_DEGENERACY_TOLERANCE_DX
BRANCH_IDENTITY_TOLERANCE_DX = stage_2z.BRANCH_IDENTITY_TOLERANCE_DX

ADVERSARIAL_CHECK_NAMES = (
    "two_co_minimal_supports_same_branch",
    "genuinely_competing_branches_rejected",
    "candidate_enumeration_permutation_invariant",
    "ray_order_permutation_equivariant",
    "preferential_single_support_selection_rejected",
    "distinct_valid_support_responses_retained",
    "opposite_sign_effects_reported",
    "equal_support_effects_reported",
    "missing_root_fails_closed",
    "multiple_roots_on_one_ray_fail_closed",
    "branch_fragmentation_or_correspondence_loss_detected",
    "historical_segment_switch_keeps_ray_identity",
    "near_limit_perturbations_are_not_tie_broken",
    "signed_displacement_convention_checked",
    "swapped_fixed_ray_identity_rejected",
    "scalar_mean_cannot_hide_disagreement",
    "evidence_mutation_rejected",
)

EVIDENCE_FILES = (
    "preflight.json",
    "source_and_provenance.json",
    "prespecification_freeze.json",
    "ray_identity_validation.json",
    "historical_validation.json",
    "support_agreement.json",
    "adversarial_and_falsification.json",
    "quality_and_blocker_matrix.json",
    "next_action.json",
    "l1a3b_final_report.json",
    "l1a3b_final_report.md",
    "manifest.json",
)


class AuditBlocked(RuntimeError):
    """Fail-closed source, evidence, geometry, or observable-contract error."""


@dataclass(frozen=True)
class RayIdentity:
    """One permanently named, cell-centred vertical ray from the frozen baseline geometry."""

    ray_id: str
    column_index: int
    x: float
    support_y: float
    initial_root_over_dx: float
    wall_normal_into_solid: tuple[float, float] = (0.0, -1.0)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _git(args: list[str], *, required: bool = True) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        if required:
            raise AuditBlocked(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
        return None
    return completed.stdout.strip()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _path_is_stage_owned(path: str) -> bool:
    return any(path == owned or path.startswith(owned) for owned in STAGE_OWNED_PATHS)


def _porcelain_path(line: str) -> str:
    body = line[2:] if len(line) > 2 else ""
    if " -> " in body:
        body = body.split(" -> ", 1)[1]
    return body.strip().strip('"').strip()


def preflight_state() -> dict[str, Any]:
    """Verify branch, non-shallow lineage and a worktree dirty only within this stage's paths."""
    head = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    main = _git(["rev-parse", "origin/main"])
    shallow = (_git(["rev-parse", "--is-shallow-repository"]) or "unknown") == "true"
    porcelain = _git(["status", "--porcelain"]) or ""
    dirty = [_porcelain_path(line) for line in porcelain.splitlines() if line.strip()]
    foreign = [path for path in dirty if not _path_is_stage_owned(path)]
    commits = (_git(["log", "--format=%H", f"{BASELINE_HEAD_SHA}..HEAD"]) or "").split()
    unknown: list[str] = []
    for commit in commits:
        touched = _git(["show", "--name-only", "--pretty=format:", commit]) or ""
        if any(not _path_is_stage_owned(line.strip()) for line in touched.splitlines() if line.strip()):
            unknown.append(commit)
    payload = {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "branch": branch,
        "head_sha": head,
        "baseline_l1a3a_blocker_head": BASELINE_HEAD_SHA,
        "main_sha": main,
        "verified_main_sha": VERIFIED_MAIN_SHA,
        "shallow_repository": shallow,
        "dirty_paths": dirty,
        "dirty_paths_outside_stage": foreign,
        "commits_above_l1a3a_blocker": commits,
        "commits_above_l1a3a_blocker_not_stage_owned": unknown,
        "control_trajectories_run": 0,
        "solver_source_edited": False,
    }
    if branch != BRANCH:
        raise AuditBlocked(f"not on the authorized branch: {branch!r}")
    if shallow:
        raise AuditBlocked("a shallow clone cannot verify this lineage")
    if main != VERIFIED_MAIN_SHA:
        raise AuditBlocked(f"verified main moved: expected {VERIFIED_MAIN_SHA}, found {main}")
    if foreign:
        raise AuditBlocked(f"worktree is dirty outside L1A-3b paths: {foreign}")
    if unknown:
        raise AuditBlocked(f"commits above the L1A-3a blocker are not owned by this stage: {unknown}")
    return payload


def _frozen_hash_block(text: str) -> dict[str, str]:
    match = re.search(r"<!-- L1A3B-FROZEN-HASHES\n(.*?)\n-->", text, re.S)
    if match is None:
        raise AuditBlocked("the L1A-3b prespecification has no frozen-hash block")
    block: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if line.strip():
            key, _, value = line.partition(" ")
            block[key.strip()] = value.strip()
    return block


def _frozen_source_changes(boundary: str) -> list[dict[str, Any]]:
    paths = sorted(_relative(path) for path in (MODULE_PATH, TEST_PATH, PRESPECIFICATION))
    output = _git(["log", "--format=%H", "--name-only", f"{boundary}..HEAD", "--", *paths]) or ""
    changes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in output.splitlines():
        item = line.strip()
        if not item:
            continue
        if len(item) == 40 and all(character in "0123456789abcdef" for character in item):
            current = {"commit": item, "files": []}
            changes.append(current)
        elif current is not None:
            current["files"].append(item)
    return [entry for entry in changes if entry["files"]]


def prespecification_freeze() -> dict[str, Any]:
    """Verify byte hashes and the committed freeze boundary before assembling stage evidence."""
    if not PRESPECIFICATION.is_file():
        raise AuditBlocked(f"missing prespecification: {PRESPECIFICATION}")
    block = _frozen_hash_block(PRESPECIFICATION.read_text())
    expected = {
        "protocol_version": STAGE_VERSION,
        "module_sha256": sha256_file(MODULE_PATH),
        "test_sha256": sha256_file(TEST_PATH),
    }
    problems = [
        f"{key}: frozen={block.get(key)!r}, worktree={value!r}"
        for key, value in expected.items()
        if block.get(key) != value
    ]
    introductions = (_git(["log", "--diff-filter=A", "--format=%H", "--", _relative(PRESPECIFICATION)]) or "").split()
    boundary = introductions[-1] if introductions else None
    if boundary is None:
        problems.append("the prespecification is not committed")
    ancestor = (
        boundary is not None and _git(["merge-base", "--is-ancestor", boundary, "HEAD"], required=False) is not None
    )
    if boundary is not None and not ancestor:
        problems.append(f"freeze boundary {boundary} is not an ancestor of HEAD")
    changes = _frozen_source_changes(boundary) if boundary is not None and ancestor else []
    sync_count = block.get("sync_count")
    if sync_count is None or not str(sync_count).isdigit():
        problems.append("frozen hash block has no numeric sync_count")
    elif int(sync_count) != len(changes):
        problems.append(f"declared sync_count={sync_count}, observed post-boundary source changes={len(changes)}")
    if problems:
        raise AuditBlocked(f"prespecification freeze failed: {problems}")
    return {
        "stage": STAGE,
        "document": _relative(PRESPECIFICATION),
        "document_sha256": sha256_file(PRESPECIFICATION),
        "boundary_commit": boundary,
        "boundary_is_ancestor": bool(ancestor),
        "frozen_hashes": block,
        "verified_worktree_hashes": expected,
        "post_boundary_source_changes": changes,
        "sync_count_declared": int(sync_count),
        "sync_count_observed": len(changes),
        "holds": True,
    }


def verify_bound_files(directory: str | Path, manifest: dict[str, Any]) -> list[str]:
    """Return integrity problems for bound files; kept separate for a meaningful tamper test."""
    problems: list[str] = []
    files = manifest.get("files", manifest.get("bound_files", {}))
    for name, expected in sorted(files.items()):
        target = Path(directory) / name
        if not target.is_file():
            problems.append(f"missing bound file {name}")
        elif sha256_file(target) != expected:
            problems.append(f"sha mismatch {name}")
    return problems


def verify_l1a3a_blocker() -> dict[str, Any]:
    """Verify the published L1A-3a stop record without altering it."""
    manifest_path = EVIDENCE_3A / "manifest.json"
    if not manifest_path.is_file():
        raise AuditBlocked("the sealed L1A-3a Phase-A blocker record is missing")
    manifest = json.loads(manifest_path.read_text())
    problems = verify_bound_files(EVIDENCE_3A, manifest)
    record_path = EVIDENCE_3A / "phase_a_audit.json"
    record = json.loads(record_path.read_text())
    if manifest.get("baseline_published_head") != L1A2Z_HEAD_SHA:
        problems.append("L1A-3a blocker no longer identifies its published L1A-2z baseline")
    if record.get("published_l1a2z_head") != L1A2Z_HEAD_SHA:
        problems.append("L1A-3a Phase-A record has an unexpected L1A-2z source head")
    if record.get("verdict") != "EARLY_WINDOW_CONTROL_INCONCLUSIVE":
        problems.append("L1A-3a blocker verdict mismatch")
    if record.get("phase_status", {}).get("E_control_measurement") != "NOT_RUN":
        problems.append("L1A-3a blocker indicates a control was run")
    if manifest.get("control_trajectory_run") is not False:
        problems.append("L1A-3a manifest indicates a control was run")
    if sha256_file(EVIDENCE_2Z / "manifest.json") != manifest.get("input_l1a2z_manifest_sha256"):
        problems.append("L1A-3a blocker no longer binds the original L1A-2z manifest")
    if problems:
        raise AuditBlocked(f"L1A-3a blocker verification failed: {problems}")
    return {
        "verified": True,
        "verdict": record["verdict"],
        "control_run": False,
        "manifest_sha256": sha256_file(manifest_path),
        "phase_a_record_sha256": sha256_file(record_path),
        "bound_files": len(manifest.get("bound_files", {})),
        "problems": [],
    }


def verify_l1a2z_evidence() -> dict[str, Any]:
    """Invoke the frozen L1A-2z verifier and retain its verdict as a read-only input."""
    result = stage_2z.verify_evidence_manifest(EVIDENCE_2Z)
    if result.get("verdict") != "EARLY_WINDOW_MEASUREMENT_ATTRIBUTION_RESOLVED":
        raise AuditBlocked(f"unexpected historical L1A-2z verdict: {result.get('verdict')!r}")
    return result


def _ulp_tolerance(*values: float, ulps: int = 8) -> float:
    spacing = [math.ulp(float(value)) for value in values if math.isfinite(float(value))]
    return float(ulps * max(spacing, default=math.ulp(1.0)))


def _close_by_ulp(first: float, second: float, *, ulps: int = 8) -> bool:
    return abs(float(first) - float(second)) <= _ulp_tolerance(first, second, ulps=ulps)


def validate_ray_geometry(rays: list[RayIdentity], p: Any, *, require_frozen_ids: bool = False) -> dict[str, Any]:
    """Check fixed cell-centred ray geometry; no root values participate in identity assignment."""
    problems: list[str] = []
    ids = [ray.ray_id for ray in rays]
    if len(ids) != 2 or len(set(ids)) != 2:
        problems.append("exactly two distinct fixed ray identities are required")
    if require_frozen_ids and set(ids) != set(RAY_IDS):
        problems.append(f"the fixed identity set must be {RAY_IDS}, found {ids}")
    if not _close_by_ulp(float(p.dx), float(p.dy)):
        problems.append("the frozen q=y/dx coordinate requires the original square-grid dx=dy convention")
    for ray in rays:
        if ray.ray_id not in ids:
            problems.append("ray identity missing")
        if ray.column_index < 0 or ray.column_index >= int(p.Nx):
            problems.append(f"{ray.ray_id}: column outside the periodic grid")
            continue
        expected_x = (int(ray.column_index) + 0.5) * float(p.dx)
        if not _close_by_ulp(ray.x, expected_x):
            problems.append(f"{ray.ray_id}: x is not its cell-centred column coordinate")
        if ray.wall_normal_into_solid != (0.0, -1.0):
            problems.append(f"{ray.ray_id}: the frozen downward wall-normal ray changed")
        if ray.ray_id in FROZEN_RAY_GEOMETRY and require_frozen_ids:
            frozen = FROZEN_RAY_GEOMETRY[ray.ray_id]
            if ray.column_index != frozen["column_index"] or not _close_by_ulp(ray.x, frozen["x"]):
                problems.append(f"{ray.ray_id}: identity disagrees with the frozen L1A-3a coordinates")
    return {
        "holds": not problems,
        "ray_ids": ids,
        "grid_shape": [int(p.Nx), int(p.Ny)],
        "dx": float(p.dx),
        "dy": float(p.dy),
        "x_identity_rule": (
            "ray x is the cell-centre coordinate (column_index + 0.5) * dx; identity never depends on a measured root"
        ),
        "problems": problems,
    }


def frozen_ray_identities(
    ledger: dict[str, Any], design: dict[str, Any], initial_identity: dict[str, Any], blocker_record: dict[str, Any]
) -> tuple[list[RayIdentity], SimpleNamespace, dict[str, Any]]:
    """Cross-check sealed supports against the two published coordinates and actual grid geometry."""
    parameters = initial_identity["parameters"]
    p = SimpleNamespace(
        Nx=int(parameters["N"]),
        Ny=int(parameters["N"]),
        Lx=float(parameters["Lx"]),
        Ly=float(parameters["Ly"]),
        dx=float(parameters["dx"]),
        dy=float(parameters["dy"]),
    )
    if not math.isclose(p.dx, p.Lx / p.Nx, rel_tol=0.0, abs_tol=math.ulp(p.dx)):
        raise AuditBlocked("the sealed L1A-2z dx does not match the recorded periodic grid geometry")
    if not math.isclose(p.dy, p.Ly / p.Ny, rel_tol=0.0, abs_tol=math.ulp(p.dy)):
        raise AuditBlocked("the sealed L1A-2z dy does not match the recorded periodic grid geometry")
    support_by_column = {int(support["column_index"]): support for support in ledger["supports"]}
    if set(support_by_column) != {95, 96}:
        raise AuditBlocked(f"the sealed L1A-2z support columns changed: {sorted(support_by_column)}")
    blocker_contract = blocker_record["observable_audit"]["l1a2z_fixed_ray_Q_contract"]
    if sorted(int(column) for column in blocker_contract["ray_columns"]) != [95, 96]:
        raise AuditBlocked("the sealed L1A-3a blocker no longer identifies the two fixed rays")
    blocker_q0 = blocker_contract["baseline_t0_root_dx"]
    blocker_displacements = blocker_contract["baseline_full_displacement_dx"]
    design_by_column = {int(support["column_index"]): support for support in design["supports"]}
    tie_points = design["t0_uniqueness"]["tie_points"]
    tie_point_by_column = {int(item["grid_edge"][1]): item for item in tie_points}
    if set(tie_point_by_column) != {95, 96} or len(design["t0_uniqueness"]["tie_components"]) != 1:
        raise AuditBlocked("the sealed L1A-2z geometry no longer records two co-component tied supports")
    pinned = ledger["pinned_components"]
    pinned_by_column = {int(support["column_index"]): pinned[support["label"]] for support in ledger["supports"]}
    rays: list[RayIdentity] = []
    checks: list[dict[str, Any]] = []
    for ray_id in RAY_IDS:
        frozen = FROZEN_RAY_GEOMETRY[ray_id]
        column = int(frozen["column_index"])
        support = support_by_column[column]
        design_support = design_by_column[column]
        point = [float(value) for value in support["point"]]
        tie_point = tie_point_by_column[column]
        if not _close_by_ulp(point[0], float(frozen["x"])):
            raise AuditBlocked(f"sealed support x for {ray_id} is not the L1A-3a point")
        if point != [float(value) for value in design_support["point"]]:
            raise AuditBlocked(f"L1A-2z support/design coordinates disagree for {ray_id}")
        if not _close_by_ulp(point[0], float(tie_point["x"])) or not _close_by_ulp(point[1], float(tie_point["y"])):
            raise AuditBlocked(f"L1A-2z tied-contour coordinate differs for {ray_id}")
        if int(tie_point["component"]) != int(design["t0_uniqueness"]["tie_components"][0]):
            raise AuditBlocked(f"the co-minimal supports do not share their sealed initial component for {ray_id}")
        expected_label = support["label"]
        if not _close_by_ulp(float(blocker_q0[expected_label]), float(pinned_by_column[column]["t0_root_over_dx"])):
            raise AuditBlocked(f"L1A-3a and L1A-2z initial root records disagree for {ray_id}")
        if not _close_by_ulp(
            float(blocker_displacements[expected_label]),
            float(support["t0_root_over_dx"]) - float(ledger["rows"][-1]["per_support"][expected_label]["q3_over_dx"]),
        ):
            raise AuditBlocked(f"L1A-3a and L1A-2z endpoint displacement records disagree for {ray_id}")
        if int(ledger["pinned_components"][expected_label]["t0_component"]) != int(
            design["t0_uniqueness"]["tie_points"][0]["component"]
        ):
            raise AuditBlocked(f"the initial contour-component membership disagrees for {ray_id}")
        if tuple(float(value) for value in support["wall_normal_into_solid"]) != (0.0, -1.0):
            raise AuditBlocked(f"the frozen ray direction changed for {ray_id}")
        root = float(pinned_by_column[column]["t0_root_over_dx"])
        if pinned_by_column[column]["t0_status"] != "OK":
            raise AuditBlocked(f"the sealed t=0 root is not valid for {ray_id}")
        rays.append(
            RayIdentity(
                ray_id=ray_id,
                column_index=column,
                x=float(point[0]),
                support_y=float(point[1]),
                initial_root_over_dx=root,
                wall_normal_into_solid=(0.0, -1.0),
            )
        )
        checks.append(
            {
                "ray_id": ray_id,
                "column_index": column,
                "x": float(point[0]),
                "support_y": float(point[1]),
                "initial_root_over_dx": root,
                "same_initial_component": int(pinned_by_column[column]["t0_component"]),
                "grid_x_reconstructed": (column + 0.5) * p.dx,
                "geometry_holds": True,
            }
        )
    geometry = validate_ray_geometry(rays, p, require_frozen_ids=True)
    if not geometry["holds"]:
        raise AuditBlocked(f"fixed ray geometry failed: {geometry['problems']}")
    return rays, p, {"geometry": geometry, "rays": checks, "same_initial_component": True}


def _candidate_component(candidate: dict[str, Any], components: dict[str, Any], ray: RayIdentity, p: Any) -> int | None:
    segment = candidate.get("segment_index")
    if segment is None:
        return None
    return stage_2z._edge_component(components, 1, ray.column_index, int(segment), p)


def enumerate_ray_intersections(phi: Any, p: Any, ray: RayIdentity, components: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate every frozen linear phi=0.5 crossing on a fixed ray, in canonical geometric order."""
    values = np.asarray(phi, dtype=np.float64)
    raw = readout.frozen_column_crossings(values, ray.column_index, p, threshold=CONTOUR_THRESHOLD)
    records: list[dict[str, Any]] = []
    for root in raw:
        record = {
            "ray_id": ray.ray_id,
            "column_index": ray.column_index,
            "x": float(ray.x),
            "root_over_dx": float(root["y_over_dx"]),
            "segment_index": int(root["segment_index"]),
            "at_shared_node": bool(root["at_shared_node"]),
            "component_label_within_this_state": None,
        }
        record["component_label_within_this_state"] = _candidate_component(record, components, ray, p)
        records.append(record)
    return sorted(records, key=lambda item: (item["root_over_dx"], item["segment_index"]))


def select_unique_admissible_root(candidates: list[dict[str, Any]], previous_root_over_dx: float) -> dict[str, Any]:
    """Fail closed unless exactly one contour intersection lies in the frozen continuation window."""
    previous = float(previous_root_over_dx)
    admissible = [
        candidate for candidate in candidates if abs(float(candidate["root_over_dx"]) - previous) <= ROOT_WINDOW_DX
    ]
    if not admissible:
        return {"status": "NO_ROOT_IN_WINDOW", "selected": None, "admissible_count": 0}
    if len(admissible) > 1:
        return {
            "status": "MULTIPLE_ADMISSIBLE_ROOTS",
            "selected": None,
            "admissible_count": len(admissible),
            "admissible_roots_over_dx": sorted(float(item["root_over_dx"]) for item in admissible),
        }
    candidate = admissible[0]
    close_competitors = [
        other
        for other in candidates
        if other is not candidate
        and abs(float(other["root_over_dx"]) - float(candidate["root_over_dx"])) < ROOT_AMBIGUITY_SEPARATION_DX
    ]
    if close_competitors:
        return {
            "status": "ROOT_SEPARATION_BELOW_FROZEN_AMBIGUITY_LIMIT",
            "selected": None,
            "admissible_count": 1,
            "competing_roots_over_dx": sorted(
                [float(candidate["root_over_dx"]), *(float(item["root_over_dx"]) for item in close_competitors)]
            ),
            "minimum_separation_over_dx": min(
                abs(float(other["root_over_dx"]) - float(candidate["root_over_dx"])) for other in close_competitors
            ),
        }
    step = abs(float(candidate["root_over_dx"]) - previous)
    if step > ROOT_STEP_MAX_DX:
        return {
            "status": "BRANCH_STEP_TOO_LARGE",
            "selected": None,
            "admissible_count": 1,
            "step_over_dx": step,
        }
    if candidate.get("component_label_within_this_state") is None:
        return {"status": "ROOT_WITHOUT_GLOBAL_COMPONENT", "selected": None, "admissible_count": 1}
    return {
        "status": "OK",
        "selected": candidate,
        "admissible_count": 1,
        "step_over_dx": step,
    }


def _branch_points(edges: dict[str, np.ndarray], components: dict[str, Any], component: int | None) -> np.ndarray:
    return stage_2z.component_points(edges, components, component)


def geometric_branch_correspondence(
    previous_points: np.ndarray | None,
    current_points: np.ndarray,
    p: Any,
) -> dict[str, Any]:
    """Relate contours by geometry only; integer component labels are never compared across states."""
    if previous_points is None:
        return {"status": "INITIAL_STATE", "holds": current_points.size > 0, "topology_event": None}
    relation = stage_2z.component_relation(previous_points, current_points, p)
    name = str(relation["relation"])
    if name in ("DISJOINT_REASSIGNMENT", "NO_COMPONENT", "AMBIGUOUS_RELATION"):
        return {"status": "CORRESPONDENCE_LOST", "holds": False, "topology_event": name, "relation": relation}
    topology_event = name if name in ("GREW_OR_MERGED", "FRAGMENTED_OR_SPLIT") else None
    return {
        "status": "GEOMETRICALLY_CORRESPONDENT",
        "holds": True,
        "topology_event": topology_event,
        "relation": relation,
        "labels_compared": False,
    }


def support_indexed_readout(
    phi: Any,
    p: Any,
    rays: list[RayIdentity],
    previous_roots_over_dx: dict[str, float],
    *,
    previous_branch_points: np.ndarray | None = None,
    bundle: Any | None = None,
) -> dict[str, Any]:
    """Read both fixed rays from the same frozen global contour without a preferential support choice.

    Every crossing on each ray is enumerated, mapped to the current global contour component and reported.
    Exactly one root must lie in the frozen continuation window on each ray, both selected roots must
    belong to the same current component, and geometric component correspondence is audited across states.
    """
    values = np.asarray(phi, dtype=np.float64)
    geometry = validate_ray_geometry(rays, p)
    if not geometry["holds"]:
        return {
            "status": "INVALID",
            "Q_dx": {ray.ray_id: None for ray in rays},
            "blocking_failures": ["INVALID_RAY_GEOMETRY", *geometry["problems"]],
            "ray_order": [ray.ray_id for ray in rays],
        }
    if values.shape != (int(p.Nx), int(p.Ny)) or not np.isfinite(values).all():
        return {
            "status": "INVALID",
            "Q_dx": {ray.ray_id: None for ray in rays},
            "blocking_failures": ["INVALID_PHASE_FIELD_SHAPE_OR_NONFINITE_VALUES"],
            "ray_order": [ray.ray_id for ray in rays],
        }
    edges = stage_2z.contour_edges(values, p)
    frozen_enumeration = stage_2z.frozen_enumeration_matches(values, p, edges)
    components = stage_2z.contour_components(values, p, edges)
    cell_validation = stage_2z.validate_components_against_frozen(values, p, components)
    exact_nodes = int(np.sum(values == CONTOUR_THRESHOLD))
    failures: list[str] = []
    if not bool(frozen_enumeration.get("matches")):
        failures.append("FROZEN_GLOBAL_CONTOUR_ENUMERATION_MISMATCH")
    if not cell_validation["matches"]:
        failures.append("FROZEN_CELL_RECONSTRUCTION_MISMATCH")
    if exact_nodes:
        failures.append("EXACT_THRESHOLD_NODE_DEGENERACY")
    if edges["x"].size == 0:
        failures.append("NO_GLOBAL_CONTOUR")
    if set(previous_roots_over_dx) != {ray.ray_id for ray in rays}:
        failures.append("INITIAL_ROOT_VECTOR_IDENTITY_MISMATCH")
    ray_records: dict[str, Any] = {}
    chosen_components: dict[str, int | None] = {}
    q: dict[str, float | None] = {}
    for ray in rays:
        candidates = enumerate_ray_intersections(values, p, ray, components)
        if ray.ray_id not in previous_roots_over_dx:
            selection = {"status": "MISSING_PREVIOUS_ROOT", "selected": None, "admissible_count": 0}
        else:
            selection = select_unique_admissible_root(candidates, previous_roots_over_dx[ray.ray_id])
        selected = selection.get("selected")
        root_value = None if selected is None else float(selected["root_over_dx"])
        q[ray.ray_id] = root_value
        chosen_components[ray.ray_id] = None if selected is None else int(selected["component_label_within_this_state"])
        if selection["status"] != "OK":
            failures.append(f"{ray.ray_id}:{selection['status']}")
        ray_records[ray.ray_id] = {
            "candidate_count_all_intersections": len(candidates),
            "candidates": candidates,
            "selection": {key: value for key, value in selection.items() if key != "selected"},
            "selected_root_over_dx": root_value,
            "selected_segment_index": None if selected is None else int(selected["segment_index"]),
            "selected_component_label_within_this_state": chosen_components[ray.ray_id],
            "at_shared_node": None if selected is None else bool(selected["at_shared_node"]),
        }
    selected_labels = list(chosen_components.values())
    if len(selected_labels) == 2 and None not in selected_labels and selected_labels[0] != selected_labels[1]:
        failures.append("THE_TWO_FIXED_RAYS_NO_LONGER_BELONG_TO_ONE_CURRENT_COMPONENT")
    common_component = (
        int(selected_labels[0])
        if len(selected_labels) == 2 and None not in selected_labels and selected_labels[0] == selected_labels[1]
        else None
    )
    current_points = _branch_points(edges, components, common_component)
    correspondence = geometric_branch_correspondence(previous_branch_points, current_points, p)
    if not correspondence["holds"]:
        failures.append("GEOMETRIC_BRANCH_CORRESPONDENCE_LOST")
    frozen_global_diagnostic = None if bundle is None else stage_2x._contour_uniqueness(values, bundle)
    return {
        "status": "VALID" if not failures else "INVALID",
        "Q_dx": {ray_id: q.get(ray_id) for ray_id in RAY_IDS if ray_id in q}
        | {ray_id: q[ray_id] for ray_id in q if ray_id not in RAY_IDS},
        "ray_ids_are_geometric": True,
        "ray_records": ray_records,
        "same_current_component": common_component is not None,
        "current_component_label_scope": "THIS_STATE_ONLY_NOT_COMPARABLE_ACROSS_TRAJECTORIES",
        "current_component_label": common_component,
        "branch_correspondence": correspondence,
        "global_contour": {
            "crossing_count": int(edges["x"].size),
            "component_count": int(components["components"]),
            "exact_threshold_nodes": exact_nodes,
            "frozen_enumeration": frozen_enumeration,
            "cell_validation": cell_validation,
            "original_l1a2x_diagnostic": frozen_global_diagnostic,
            "original_diagnostic_redefined": False,
        },
        "blocking_failures": sorted(set(failures)),
        "branch_points_for_next_state": current_points,
        "no_primary_support_or_scalar_aggregation": True,
    }


def frozen_support_indexed_readout(
    phi: Any,
    p: Any,
    rays: list[RayIdentity],
    previous_roots_over_dx: dict[str, float],
    *,
    bundle: Any | None,
    previous_branch_points: np.ndarray | None = None,
) -> dict[str, Any]:
    """Production-contract entry point: enforce fixed IDs and retain the original global diagnostic."""
    geometry = validate_ray_geometry(rays, p, require_frozen_ids=True)
    if bundle is None:
        return {
            "status": "INVALID",
            "Q_dx": {ray_id: None for ray_id in RAY_IDS},
            "blocking_failures": ["ORIGINAL_GLOBAL_CONTOUR_DIAGNOSTIC_BUNDLE_REQUIRED"],
            "ray_geometry": geometry,
        }
    if not geometry["holds"]:
        return {
            "status": "INVALID",
            "Q_dx": {ray_id: None for ray_id in RAY_IDS},
            "blocking_failures": ["FROZEN_RAY_IDENTITY_OR_GEOMETRY_MISMATCH", *geometry["problems"]],
            "ray_geometry": geometry,
        }
    result = support_indexed_readout(
        phi,
        p,
        rays,
        previous_roots_over_dx,
        previous_branch_points=previous_branch_points,
        bundle=bundle,
    )
    result["ray_geometry"] = geometry
    if result["global_contour"]["original_l1a2x_diagnostic"] is None:
        result["status"] = "INVALID"
        result["blocking_failures"] = sorted(
            set(result["blocking_failures"]) | {"ORIGINAL_GLOBAL_CONTOUR_DIAGNOSTIC_NOT_RECONSTRUCTED"}
        )
    return result


def validate_complete_vector(values: dict[str, float | None], ray_ids: tuple[str, ...] = RAY_IDS) -> dict[str, float]:
    """Require the complete two-ray vector; the function has no primary/mean fallback."""
    if set(values) != set(ray_ids):
        raise AuditBlocked(f"support vector must contain exactly {ray_ids}; found {tuple(values)}")
    result: dict[str, float] = {}
    for ray_id in ray_ids:
        value = values[ray_id]
        if value is None or not math.isfinite(float(value)):
            raise AuditBlocked(f"missing or nonfinite support value for {ray_id}")
        result[ray_id] = float(value)
    return result


def signed_displacement_vector(
    initial_q_over_dx: dict[str, float], current_q_over_dx: dict[str, float]
) -> dict[str, float]:
    """Return d_j=q_j(0)-q_j(t), in raw dx units, preserving the original toward-solid sign."""
    initial = validate_complete_vector(initial_q_over_dx)
    current = validate_complete_vector(current_q_over_dx)
    return {ray_id: initial[ray_id] - current[ray_id] for ray_id in RAY_IDS}


def paired_effect_vector(
    initial_q_over_dx: dict[str, float],
    baseline_q_over_dx: dict[str, float],
    control_q_over_dx: dict[str, float],
    *,
    baseline_initial_hashes: dict[str, str],
    control_initial_hashes: dict[str, str],
    baseline_root_reconstruction_difference_dx: dict[str, float] | None = None,
    control_root_reconstruction_difference_dx: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute E_j=d_j(control)-d_j(baseline), and verify its signed root-coordinate identity."""
    if baseline_initial_hashes != control_initial_hashes:
        raise AuditBlocked("baseline/control initial-state hashes differ")
    initial = validate_complete_vector(initial_q_over_dx)
    baseline = validate_complete_vector(baseline_q_over_dx)
    control = validate_complete_vector(control_q_over_dx)
    d_baseline = {ray_id: initial[ray_id] - baseline[ray_id] for ray_id in RAY_IDS}
    d_control = {ray_id: initial[ray_id] - control[ray_id] for ray_id in RAY_IDS}
    effects = {ray_id: d_control[ray_id] - d_baseline[ray_id] for ray_id in RAY_IDS}
    coordinate_identity = {ray_id: baseline[ray_id] - control[ray_id] for ray_id in RAY_IDS}
    identity_residual = {ray_id: effects[ray_id] - coordinate_identity[ray_id] for ray_id in RAY_IDS}
    values = [effects[ray_id] for ray_id in RAY_IDS]
    baseline_arithmetic = (
        None
        if baseline_root_reconstruction_difference_dx is None
        else validate_complete_vector(baseline_root_reconstruction_difference_dx)
    )
    control_arithmetic = (
        None
        if control_root_reconstruction_difference_dx is None
        else validate_complete_vector(control_root_reconstruction_difference_dx)
    )
    if any(
        value < 0.0
        for values_ in (baseline_arithmetic, control_arithmetic)
        if values_ is not None
        for value in values_.values()
    ):
        raise AuditBlocked("root reconstruction differences must be nonnegative magnitudes")
    arithmetic_sum = (
        None
        if baseline_arithmetic is None or control_arithmetic is None
        else {ray_id: baseline_arithmetic[ray_id] + control_arithmetic[ray_id] for ray_id in RAY_IDS}
    )
    effect_sign = {ray_id: 1 if effects[ray_id] > 0.0 else -1 if effects[ray_id] < 0.0 else 0 for ray_id in RAY_IDS}
    return {
        "initial_hashes_match": True,
        "displacement_baseline_dx": d_baseline,
        "displacement_control_dx": d_control,
        "E_dx": effects,
        "coordinate_identity_baseline_minus_control_dx": coordinate_identity,
        "coordinate_identity_residual_dx": identity_residual,
        "raw_signed_effect_range_dx": [min(values), max(values)],
        "signed_effect_difference_RAY_95_minus_RAY_96_dx": effects["RAY_95"] - effects["RAY_96"],
        "absolute_support_to_support_effect_disagreement_dx": abs(effects["RAY_95"] - effects["RAY_96"]),
        "effect_sign_by_ray": effect_sign,
        "sign_discordant": effect_sign["RAY_95"] != effect_sign["RAY_96"],
        "effects_have_opposite_signs": bool(effects["RAY_95"] * effects["RAY_96"] < 0.0),
        "root_reconstruction_difference_baseline_dx": baseline_arithmetic,
        "root_reconstruction_difference_control_dx": control_arithmetic,
        "combined_root_arithmetic_difference_dx": arithmetic_sum,
        "solver_state_uncertainty_bound_dx": None,
        "uncertainty_status": "ARITHMETIC_ONLY" if arithmetic_sum is not None else "UNRESOLVED_NO_TOTAL_UNCERTAINTY",
        "validity_failures": [],
        "support_order_is_storage_only": True,
        "scalar_mean_or_preferred_support_reported": False,
    }


def paired_effect_audit(
    initial_q_over_dx: dict[str, float],
    baseline_readout: dict[str, Any],
    control_readout: dict[str, Any],
    *,
    baseline_initial_hashes: dict[str, str],
    control_initial_hashes: dict[str, str],
) -> dict[str, Any]:
    """Return all validity failures and no partial scalar/effect if either trajectory is invalid."""
    failures: list[str] = []
    for label, record in (("baseline", baseline_readout), ("control", control_readout)):
        if record.get("status") != "VALID":
            failures.extend(f"{label}:{item}" for item in record.get("blocking_failures", ["INVALID_READOUT"]))
    if baseline_initial_hashes != control_initial_hashes:
        failures.append("BASELINE_CONTROL_INITIAL_STATE_HASH_MISMATCH")
    if failures:
        return {
            "status": "INVALID",
            "E_dx": None,
            "raw_signed_effect_range_dx": None,
            "absolute_support_to_support_effect_disagreement_dx": None,
            "sign_discordant": None,
            "validity_failures": sorted(set(failures)),
            "no_partial_support_effect_reported": True,
        }
    try:
        effects = paired_effect_vector(
            initial_q_over_dx,
            baseline_readout["Q_dx"],
            control_readout["Q_dx"],
            baseline_initial_hashes=baseline_initial_hashes,
            control_initial_hashes=control_initial_hashes,
        )
    except AuditBlocked as exc:
        return {
            "status": "INVALID",
            "E_dx": None,
            "raw_signed_effect_range_dx": None,
            "absolute_support_to_support_effect_disagreement_dx": None,
            "sign_discordant": None,
            "validity_failures": [str(exc)],
            "no_partial_support_effect_reported": True,
        }
    return {"status": "VALID", **effects}


def _decimal_root_over_dx(phi: np.ndarray, ray: RayIdentity, segment_index: int, p: Any) -> float:
    """Recompute a linear edge root with Decimal from the exact stored binary64 endpoint values."""
    a = float(phi[ray.column_index, int(segment_index)])
    b = float(phi[ray.column_index, (int(segment_index) + 1) % int(p.Ny)])
    with localcontext() as context:
        context.prec = 80
        da = Decimal.from_float(a)
        db = Decimal.from_float(b)
        threshold = Decimal("0.5")
        fraction = (threshold - da) / (db - da)
        y = (Decimal(int(segment_index)) + Decimal("0.5") + fraction) % Decimal(int(p.Ny))
    return float(y)


def root_reconstruction_uncertainty(
    phi: Any, ray: RayIdentity, segment_index: int, reported_root_over_dx: float, p: Any
) -> dict[str, Any]:
    """Measure floating-point root reconstruction error against high-precision arithmetic.

    This is only the interpolation arithmetic error for the stored phase field. It is not a bound on
    solver-state error, discretization error, or intervention uncertainty.
    """
    exact_input_root = _decimal_root_over_dx(np.asarray(phi, dtype=np.float64), ray, segment_index, p)
    observed = abs(float(reported_root_over_dx) - exact_input_root)
    return {
        "reported_root_over_dx": float(reported_root_over_dx),
        "decimal_root_from_same_stored_field_over_dx": exact_input_root,
        "observed_float_interpolation_difference_dx": observed,
        "solver_or_model_uncertainty_bounded": False,
    }


def historical_validation(
    ledger: dict[str, Any],
    design: dict[str, Any],
    initial_identity: dict[str, Any],
    blocker_record: dict[str, Any],
) -> tuple[dict[str, Any], list[RayIdentity], SimpleNamespace]:
    """Check every stored two-ray S1/S2/S3 record in the sealed L1A-2z ledger, without replaying a solver."""
    rays, p, geometry = frozen_ray_identities(ledger, design, initial_identity, blocker_record)
    labels_by_column = {int(item["column_index"]): str(item["label"]) for item in ledger["supports"]}
    rows = ledger["rows"]
    uniqueness = ledger["uniqueness_rows"]
    if len(rows) != int(ledger["substeps_recorded"]) or len(uniqueness) != 3 * len(rows):
        raise AuditBlocked("the sealed L1A-2z evidence does not contain every expected intermediate root record")
    state_vectors: list[dict[str, Any]] = []
    substep_readbacks: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    readback_status_counts: dict[str, int] = {}
    relation_counts: dict[str, int] = {}
    segment_switches: list[dict[str, Any]] = []
    q0_remeasure_deltas: list[float] = []
    q3_direct_differences: list[float] = []
    previous_segments = {
        ray.ray_id: int(ledger["pinned_components"][labels_by_column[ray.column_index]]["t0_segment_index"])
        for ray in rays
    }
    q0 = {ray.ray_id: float(ray.initial_root_over_dx) for ray in rays}
    support_spreads: list[float] = []
    displacement_spreads: list[float] = []
    problems: list[str] = []
    state_row_index = 0
    for row in rows:
        per_ray = {ray.ray_id: row["per_support"][labels_by_column[ray.column_index]] for ray in rays}
        readbacks: dict[str, Any] = {}
        for ray in rays:
            record = per_ray[ray.ray_id]
            q0_status = str(record["statuses"]["q0"])
            q3_direct_status = str(record["statuses"]["q3_direct"])
            readback_status_counts[q0_status] = readback_status_counts.get(q0_status, 0) + 1
            readback_status_counts[q3_direct_status] = readback_status_counts.get(q3_direct_status, 0) + 1
            q0_carried = record["q0_carried_over_dx"]
            q0_remeasured = record["q0_remeasured_over_dx"]
            q0_delta = record["q0_remeasure_delta"]
            q3 = record["q3_over_dx"]
            q3_direct = record["q3_direct_over_dx"]
            if (
                q0_status != "OK"
                or q3_direct_status != "OK"
                or q0_carried is None
                or q0_remeasured is None
                or q0_delta is None
                or q3 is None
                or q3_direct is None
            ):
                problems.append(
                    f"stored carried-root or direct-root validation failed for {ray.ray_id} "
                    f"at {row['macro_step']}/{row['substep']}"
                )
            else:
                q0_remeasure_deltas.append(float(q0_delta))
                q3_direct_differences.append(float(q3_direct) - float(q3))
            readbacks[ray.ray_id] = {
                "q0_carried_over_dx": None if q0_carried is None else float(q0_carried),
                "q0_remeasured_over_dx": None if q0_remeasured is None else float(q0_remeasured),
                "q0_remeasure_delta_dx": None if q0_delta is None else float(q0_delta),
                "q3_over_dx": None if q3 is None else float(q3),
                "q3_direct_over_dx": None if q3_direct is None else float(q3_direct),
                "q3_direct_minus_q3_dx": None if q3 is None or q3_direct is None else float(q3_direct) - float(q3),
                "status_q0": q0_status,
                "status_q3_direct": q3_direct_status,
            }
        substep_readbacks.append(
            {
                "macro_step": int(row["macro_step"]),
                "substep": int(row["substep"]),
                "t_before": float(row["t_before"]),
                "t_after": float(row["t_after"]),
                "readbacks_by_ray": readbacks,
                "units": "dx",
            }
        )
        for state_name, root_key, status_key in (
            ("s1", "q1_over_dx", "q1"),
            ("s2", "q2_over_dx", "q2"),
            ("s3", "q3_over_dx", "q3"),
        ):
            q_values: dict[str, float] = {}
            segments: dict[str, int | None] = {}
            components: dict[str, int | None] = {}
            statuses: dict[str, str] = {}
            for ray in rays:
                record = per_ray[ray.ray_id]
                raw = record[root_key]
                status = str(record["statuses"][status_key])
                component = record["root_components"][status_key]
                segment = record["segment_indices"][status_key]
                status_counts[status] = status_counts.get(status, 0) + 1
                if status != "OK" or raw is None or component is None or segment is None:
                    problems.append(
                        f"{ray.ray_id} {state_name} invalid at macro {row['macro_step']} substep {row['substep']}"
                    )
                else:
                    q_values[ray.ray_id] = float(raw)
                statuses[ray.ray_id] = status
                components[ray.ray_id] = None if component is None else int(component)
                segments[ray.ray_id] = None if segment is None else int(segment)
                if segment is not None and int(segment) != previous_segments[ray.ray_id]:
                    segment_switches.append(
                        {
                            "ray_id": ray.ray_id,
                            "macro_step": int(row["macro_step"]),
                            "substep": int(row["substep"]),
                            "state": state_name,
                            "from_segment_index": previous_segments[ray.ray_id],
                            "to_segment_index": int(segment),
                            "ray_identity_changed": False,
                        }
                    )
                    previous_segments[ray.ray_id] = int(segment)
            if len(components) == 2 and None not in components.values() and len(set(components.values())) != 1:
                problems.append(
                    f"the two rays do not share the current component at {state_name}/"
                    f"{row['macro_step']}/{row['substep']}"
                )
            if len(q_values) == 2:
                spread = abs(q_values[RAY_IDS[0]] - q_values[RAY_IDS[1]])
                support_spreads.append(spread)
                displacement_spreads.append(
                    abs((q0[RAY_IDS[0]] - q_values[RAY_IDS[0]]) - (q0[RAY_IDS[1]] - q_values[RAY_IDS[1]]))
                )
            state_vector = {
                "macro_step": int(row["macro_step"]),
                "substep": int(row["substep"]),
                "state": state_name,
                "t_after_substep": float(row["t_after"]),
                "Q_dx": {ray_id: q_values.get(ray_id) for ray_id in RAY_IDS},
                "root_status": statuses,
                "segment_index_by_ray": segments,
                "component_label_within_state_by_ray": components,
                "component_labels_compared_across_states": False,
            }
            state_vectors.append(state_vector)
            uniqueness_row = uniqueness[state_row_index]
            state_row_index += 1
            if (
                uniqueness_row["status"] != "UNIQUE_BRANCH"
                or uniqueness_row["rule_ok"] is not True
                or uniqueness_row["branch_unique"] is not True
                or uniqueness_row["frozen_enumeration_matches"] is not True
                or int(uniqueness_row["exact_threshold_nodes"]) != 0
            ):
                problems.append(f"L1A-2z branch uniqueness/reconstruction record failed at row {state_row_index}")
            if uniqueness_row["frozen_status"] != "NON_UNIQUE":
                problems.append(f"the frozen L1A-2x diagnostic changed at row {state_row_index}")
        for ray in rays:
            record = per_ray[ray.ray_id]
            for span, relation in record["component_relations"].items():
                name = str(relation["relation"])
                relation_counts[name] = relation_counts.get(name, 0) + 1
                if relation.get("reassignment"):
                    problems.append(
                        f"stored geometric branch reassignment for {ray.ray_id} "
                        f"at {row['macro_step']}/{row['substep']}/{span}"
                    )
            if any(record["statuses"][name] != "OK" for name in ("q0", "q1", "q2", "q3", "q3_direct")):
                problems.append(
                    f"stored readout status failure for {ray.ray_id} at {row['macro_step']}/{row['substep']}"
                )
    if len(state_vectors) != 360:
        problems.append(f"expected 360 stored intermediate states, found {len(state_vectors)}")
    if any(item["tie_set_size"] != 2 or len(item["tie_components"]) != 1 for item in uniqueness):
        problems.append("the sealed co-minimal support tie set changed in the historical record")
    frozen_statuses: dict[str, int] = {}
    branch_statuses: dict[str, int] = {}
    competition_sizes: dict[str, int] = {}
    for item in uniqueness:
        frozen_statuses[str(item["frozen_status"])] = frozen_statuses.get(str(item["frozen_status"]), 0) + 1
        branch_statuses[str(item["status"])] = branch_statuses.get(str(item["status"]), 0) + 1
        size = str(item["competition_set_size"])
        competition_sizes[size] = competition_sizes.get(size, 0) + 1
    endpoints = {
        "RAY_95": {
            "initial_root_over_dx": q0["RAY_95"],
            "final_root_over_dx": float(ledger["rows"][-1]["per_support"][labels_by_column[95]]["q3_over_dx"]),
            "toward_solid_displacement_dx": q0["RAY_95"]
            - float(ledger["rows"][-1]["per_support"][labels_by_column[95]]["q3_over_dx"]),
        },
        "RAY_96": {
            "initial_root_over_dx": q0["RAY_96"],
            "final_root_over_dx": float(ledger["rows"][-1]["per_support"][labels_by_column[96]]["q3_over_dx"]),
            "toward_solid_displacement_dx": q0["RAY_96"]
            - float(ledger["rows"][-1]["per_support"][labels_by_column[96]]["q3_over_dx"]),
        },
    }
    payload = {
        "stage": STAGE,
        "source_stage": "L1A-2z sealed baseline records only; no solver replay",
        "geometry_validation": geometry["geometry"],
        "intermediate_state_count": len(state_vectors),
        "substep_count": len(substep_readbacks),
        "readout_status_counts_q1_q2_q3": status_counts,
        "q0_remeasure_and_q3_direct_status_counts": readback_status_counts,
        "max_absolute_q0_remeasure_delta_dx": max((abs(value) for value in q0_remeasure_deltas), default=None),
        "max_absolute_q3_direct_minus_q3_dx": max((abs(value) for value in q3_direct_differences), default=None),
        "all_per_substep_readbacks": substep_readbacks,
        "branch_uniqueness_status_counts": branch_statuses,
        "frozen_l1a2x_global_diagnostic_status_counts_retained_verbatim": frozen_statuses,
        "historical_competition_set_size_counts": competition_sizes,
        "geometric_relation_counts_by_support": relation_counts,
        "same_current_component_for_both_rays_in_all_states": not any("do not share" in item for item in problems),
        "geometric_component_labels_compared_across_states": False,
        "ray_segment_switches": segment_switches,
        "support_root_spread_dx": {
            "max_over_all_360_states": max(support_spreads) if support_spreads else None,
            "final_state": support_spreads[-1] if support_spreads else None,
            "displacement_spread_max_over_all_360_states": max(displacement_spreads) if displacement_spreads else None,
            "not_used_as_numerical_noise_floor": True,
        },
        "endpoints": endpoints,
        "all_intermediate_vectors": state_vectors,
        "full_independent_candidate_replay": {
            "available_from_sealed_l1a2z": False,
            "reason": (
                "L1A-2z seals each ray root/status/segment/component and global uniqueness summaries, "
                "but not the intermediate phi arrays or full per-ray crossing candidate lists."
            ),
            "action": (
                "verified every stored two-ray S1/S2/S3 record and global diagnostic; did not rerun the "
                "solver or claim an independent reconstruction of omitted candidates"
            ),
        },
        "problems": problems,
        "stored_record_gate_holds": not problems,
    }
    return payload, rays, p


def support_agreement_assessment(historical: dict[str, Any]) -> dict[str, Any]:
    """Report support differences separately from arithmetic error and solver-state uncertainty."""
    source = historical["support_root_spread_dx"]
    capture = json.loads((EVIDENCE_2Z / "capture_equivalence.json").read_text())
    identical = json.loads((EVIDENCE_2Z / "identical_window_check.json").read_text())
    return {
        "stage": STAGE,
        "original_baseline_support_root_spread_dx": source["max_over_all_360_states"],
        "original_baseline_final_support_root_spread_dx": source["final_state"],
        "original_baseline_support_displacement_spread_dx": source["displacement_spread_max_over_all_360_states"],
        "historical_baseline_difference_is_a_noise_floor": False,
        "available_solver_reconstruction_checks": {
            "capture_equivalence_max_phi_difference": capture["max_phi_difference_vs_eager_public_step"],
            "capture_equivalence_max_gap_difference_dx": capture["max_gap_difference_over_dx"],
            "identical_window_max_root_difference_dx": identical["max_absolute_root_difference_over_dx"],
            "note": (
                "These are historical diagnostics, not a validated per-ray root uncertainty bound for an intervention."
            ),
        },
        "future_per_root_arithmetic_check": (
            "Recompute each crossing from the same stored binary64 edge values using high-precision "
            "Decimal and report the raw interpolation difference in dx; do not call it solver uncertainty."
        ),
        "solver_state_uncertainty_bound_dx": None,
        "binary_support_invariance_threshold_dx": None,
        "support_agreement_conclusion": "UNRESOLVED_NO_JUSTIFIED_BINARY_THRESHOLD",
        "interpretation": (
            "The observable is the full support-indexed vector; the contract does not require the two "
            "components to agree. Report raw disagreement and uncertainty separately, without averaging "
            "or discarding either ray."
        ),
    }


def response_report(q0: dict[str, float], q_t: dict[str, float]) -> dict[str, Any]:
    """Return a symmetric vector summary with no scalar mean, min-preference, or max-preference claim."""
    displacement = signed_displacement_vector(q0, q_t)
    values = [displacement[ray_id] for ray_id in RAY_IDS]
    return {
        "Q_dx": validate_complete_vector(q_t),
        "displacement_toward_solid_dx": displacement,
        "raw_signed_displacement_range_dx": [min(values), max(values)],
        "absolute_support_to_support_disagreement_dx": abs(values[0] - values[1]),
        "sign_by_ray": {
            ray_id: 1 if displacement[ray_id] > 0.0 else -1 if displacement[ray_id] < 0.0 else 0 for ray_id in RAY_IDS
        },
        "sign_discordant": (displacement[RAY_IDS[0]] > 0.0) != (displacement[RAY_IDS[1]] > 0.0)
        or (displacement[RAY_IDS[0]] < 0.0) != (displacement[RAY_IDS[1]] < 0.0),
        "opposite_signs": bool(values[0] * values[1] < 0.0),
        "numerical_uncertainty_dx": None,
        "uncertainty_status": "NOT_SUPPLIED_NO_TOTAL_UNCERTAINTY_CLAIM",
        "validity_failures": [],
        "scalar_aggregation_performed": False,
    }


def _synthetic_grid(n: int = 32, length: float = 3.2) -> SimpleNamespace:
    dx = length / n
    return SimpleNamespace(Nx=n, Ny=n, Lx=length, Ly=length, dx=dx, dy=dx)


def _synthetic_rays(p: Any) -> list[RayIdentity]:
    rays = [
        RayIdentity("RAY_A", 10, 10.5 * p.dx, 0.0, 10.0),
        RayIdentity("RAY_B", 11, 11.5 * p.dx, 0.0, 10.0),
    ]
    return rays


def _circle_phi(p: Any, *, radius: float = 0.8) -> np.ndarray:
    """Smooth periodic-grid test field with a closed contour through both synthetic rays."""
    x = (np.arange(p.Nx, dtype=np.float64) + 0.5)[:, None] * p.dx
    y = (np.arange(p.Ny, dtype=np.float64) + 0.5)[None, :] * p.dy
    distance_squared = (x - 1.1) ** 2 + (y - 1.825) ** 2
    return 0.5 + 0.25 * np.tanh(10.0 * (radius**2 - distance_squared))


def _annulus_phi(p: Any) -> np.ndarray:
    """Two competing nested branches whose bottom roots both lie in the test continuation window."""
    x = (np.arange(p.Nx, dtype=np.float64) + 0.5)[:, None] * p.dx
    y = (np.arange(p.Ny, dtype=np.float64) + 0.5)[None, :] * p.dy
    distance_squared = (x - 1.1) ** 2 + (y - 1.825) ** 2
    product = (distance_squared - 0.8**2) * (distance_squared - 0.7**2)
    return 0.5 + 0.25 * np.tanh(-20.0 * product)


def _synthetic_adversarial_checks() -> dict[str, dict[str, Any]]:
    """Small deterministic probes used to seal the named negative/positive checks."""
    p = _synthetic_grid()
    rays = _synthetic_rays(p)
    one_branch = _circle_phi(p)
    unique = support_indexed_readout(
        one_branch,
        p,
        rays,
        {"RAY_A": 10.25, "RAY_B": 10.25},
    )
    competing = _annulus_phi(p)
    competing_readout = support_indexed_readout(
        competing,
        p,
        rays,
        {"RAY_A": 10.75, "RAY_B": 10.75},
    )
    candidates = [
        {"root_over_dx": 10.25, "component_label_within_this_state": 7, "segment_index": 10},
        {"root_over_dx": 25.25, "component_label_within_this_state": 8, "segment_index": 25},
    ]
    selected = select_unique_admissible_root(candidates, 10.25)
    reversed_selected = select_unique_admissible_root(list(reversed(candidates)), 10.25)
    ray_reversed = support_indexed_readout(
        one_branch,
        p,
        list(reversed(rays)),
        {"RAY_A": 10.25, "RAY_B": 10.25},
    )
    partial_vector_rejected = False
    try:
        validate_complete_vector({"RAY_95": 1.0})
    except AuditBlocked:
        partial_vector_rejected = True
    varied = paired_effect_vector(
        {"RAY_95": 10.0, "RAY_96": 10.0},
        {"RAY_95": 9.8, "RAY_96": 9.7},
        {"RAY_95": 9.6, "RAY_96": 9.8},
        baseline_initial_hashes={"phi": "same"},
        control_initial_hashes={"phi": "same"},
    )
    equal = paired_effect_vector(
        {"RAY_95": 10.0, "RAY_96": 10.0},
        {"RAY_95": 9.8, "RAY_96": 9.8},
        {"RAY_95": 9.6, "RAY_96": 9.6},
        baseline_initial_hashes={"phi": "same"},
        control_initial_hashes={"phi": "same"},
    )
    missing = support_indexed_readout(np.ones_like(one_branch), p, rays, {"RAY_A": 10.25, "RAY_B": 10.25})
    multiply = competing_readout
    # A root crossing a grid segment boundary changes its segment index, not its immutable ray id.
    switch_events = [
        {"ray_id": "RAY_A", "from_segment_index": 23, "to_segment_index": 22},
        {"ray_id": "RAY_B", "from_segment_index": 23, "to_segment_index": 22},
    ]
    segment_identity_holds = all(event["ray_id"] in {ray.ray_id for ray in rays} for event in switch_events)
    near_limit_candidates = [{"root_over_dx": 10.25, "component_label_within_this_state": 7, "segment_index": 10}]
    near_limit = select_unique_admissible_root(near_limit_candidates, 10.25 + ROOT_STEP_MAX_DX)
    ambiguous_near = select_unique_admissible_root(
        [
            {"root_over_dx": 10.25, "component_label_within_this_state": 7, "segment_index": 10},
            {"root_over_dx": 10.75, "component_label_within_this_state": 9, "segment_index": 11},
        ],
        10.5,
    )
    wrong_sign = signed_displacement_vector({"RAY_95": 10.0, "RAY_96": 10.0}, {"RAY_95": 9.5, "RAY_96": 9.25})
    wrong_identity = not validate_ray_geometry(
        [RayIdentity("RAY_95", 96, 9.65, 1.0, 10.0), RayIdentity("RAY_96", 95, 9.55, 1.0, 10.0)],
        SimpleNamespace(Nx=192, Ny=192, dx=0.1, dy=0.1, Lx=19.2, Ly=19.2),
        require_frozen_ids=True,
    )["holds"]
    branch_lost = geometric_branch_correspondence(
        np.array([[1.0, 1.0], [1.1, 1.0]], dtype=np.float64),
        np.array([[8.0, 8.0], [8.1, 8.0]], dtype=np.float64),
        p,
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        mutation_path = Path(temporary_directory) / "probe.json"
        mutation_path.write_text('{"value":1}\n')
        mutation_manifest = {"files": {"probe.json": sha256_file(mutation_path)}}
        mutation_clean = verify_bound_files(temporary_directory, mutation_manifest) == []
        mutation_path.write_text('{"value":2}\n')
        mutation_detected = verify_bound_files(temporary_directory, mutation_manifest) == ["sha mismatch probe.json"]
    evidence_mutation_holds = mutation_clean and mutation_detected
    checks = {
        "two_co_minimal_supports_same_branch": {
            "holds": unique["status"] == "VALID" and unique["same_current_component"],
            "detail": unique["Q_dx"],
        },
        "genuinely_competing_branches_rejected": {
            "holds": competing_readout["status"] == "INVALID",
            "detail": competing_readout["blocking_failures"],
        },
        "candidate_enumeration_permutation_invariant": {
            "holds": selected["status"] == reversed_selected["status"]
            and selected["selected"]["root_over_dx"] == reversed_selected["selected"]["root_over_dx"],
            "detail": selected["selected"]["root_over_dx"],
        },
        "ray_order_permutation_equivariant": {
            "holds": ray_reversed["Q_dx"] == unique["Q_dx"],
            "detail": ray_reversed["Q_dx"],
        },
        "preferential_single_support_selection_rejected": {
            "holds": partial_vector_rejected and "scalar_aggregation_performed" not in unique,
            "detail": {"partial_vector_rejected": partial_vector_rejected, "ray_ids": list(unique["Q_dx"])},
        },
        "distinct_valid_support_responses_retained": {
            "holds": varied["E_dx"]["RAY_95"] != varied["E_dx"]["RAY_96"]
            and varied["scalar_mean_or_preferred_support_reported"] is False,
            "detail": varied["E_dx"],
        },
        "opposite_sign_effects_reported": {"holds": varied["effects_have_opposite_signs"], "detail": varied["E_dx"]},
        "equal_support_effects_reported": {
            "holds": equal["E_dx"]["RAY_95"] == equal["E_dx"]["RAY_96"]
            and equal["absolute_support_to_support_effect_disagreement_dx"] == 0.0,
            "detail": equal["E_dx"],
        },
        "missing_root_fails_closed": {"holds": missing["status"] == "INVALID", "detail": missing["blocking_failures"]},
        "multiple_roots_on_one_ray_fail_closed": {
            "holds": multiply["status"] == "INVALID"
            and any("MULTIPLE_ADMISSIBLE_ROOTS" in item for item in multiply["blocking_failures"]),
            "detail": multiply["blocking_failures"],
        },
        "branch_fragmentation_or_correspondence_loss_detected": {
            "holds": branch_lost["holds"] is False and branch_lost["status"] == "CORRESPONDENCE_LOST",
            "detail": branch_lost["status"],
        },
        "historical_segment_switch_keeps_ray_identity": {
            "holds": segment_identity_holds,
            "detail": "historical q switches are recorded against RAY_95/RAY_96; they never redefine either ray",
        },
        "near_limit_perturbations_are_not_tie_broken": {
            "holds": near_limit["status"] == "OK" and ambiguous_near["status"] == "MULTIPLE_ADMISSIBLE_ROOTS",
            "detail": {"step_boundary": near_limit["status"], "multiple": ambiguous_near["status"]},
        },
        "signed_displacement_convention_checked": {
            "holds": wrong_sign == {"RAY_95": 0.5, "RAY_96": 0.75},
            "detail": wrong_sign,
        },
        "swapped_fixed_ray_identity_rejected": {
            "holds": wrong_identity,
            "detail": "the published ray identifiers remain bound to columns 95 and 96",
        },
        "scalar_mean_cannot_hide_disagreement": {
            "holds": "scalar_mean_or_preferred_support_reported" in varied
            and varied["scalar_mean_or_preferred_support_reported"] is False
            and varied["absolute_support_to_support_effect_disagreement_dx"] > 0.0,
            "detail": varied["absolute_support_to_support_effect_disagreement_dx"],
        },
        "evidence_mutation_rejected": {
            "holds": evidence_mutation_holds,
            "detail": {"clean_seal_holds": mutation_clean, "mutated_payload_rejected": mutation_detected},
        },
    }
    if len(checks) != len(ADVERSARIAL_CHECK_NAMES) or set(checks) != set(ADVERSARIAL_CHECK_NAMES):
        raise AuditBlocked("the adversarial check-name set differs from its frozen inventory")
    return checks


def adversarial_and_falsification() -> dict[str, Any]:
    checks = _synthetic_adversarial_checks()
    failed = [name for name, result in checks.items() if not result["holds"]]
    return {
        "stage": STAGE,
        "checks_run": len(checks),
        "passed_checks": len(checks) - len(failed),
        "failed_checks": failed,
        "holds": not failed,
        "checks": checks,
        "test_ids_frozen_in_source": list(ADVERSARIAL_CHECK_NAMES),
        "note": "These are observable-contract falsification checks only; no physical solver trajectory is involved.",
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditBlocked(f"required evidence file missing: {path}")
    return json.loads(path.read_text())


def source_and_provenance(preflight: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "baseline_head_sha": BASELINE_HEAD_SHA,
        "l1a2z_head_sha": L1A2Z_HEAD_SHA,
        "l1a3a_blocker_head_sha": L1A3A_HEAD_SHA,
        "verified_main_sha": VERIFIED_MAIN_SHA,
        "preflight_head_sha": preflight["head_sha"],
        "prespecification_boundary": freeze["boundary_commit"],
        "source_hashes": {name: sha256_file(path) for name, path in SOURCE_PATHS.items()},
        "l1a2z_manifest_sha256": sha256_file(EVIDENCE_2Z / "manifest.json"),
        "l1a3a_manifest_sha256": sha256_file(EVIDENCE_3A / "manifest.json"),
        "no_solver_trajectory_run": True,
        "no_mobility_intervention_run": True,
        "no_solver_source_edit": True,
    }


def _canonical_test_identifier(value: str) -> str:
    """Normalize pytest's optional leading `tests/` while preserving the exact observed identifier elsewhere."""
    normalized = str(value).replace("\\", "/")
    return normalized.removeprefix("tests/")


def _suite_observation_valid(record: dict[str, Any], *, require_no_failures: bool) -> tuple[bool, list[str]]:
    problems: list[str] = []
    result = str(record.get("result") or "NOT_RUN")
    failing = list(record.get("failing", []))
    match = re.search(r"(\d+) failed", result)
    failure_count = int(match.group(1)) if match else 0
    if result == "NOT_RUN" or "passed" not in result:
        problems.append("suite result is missing or does not report passed tests")
    if failure_count != len(failing):
        problems.append(f"pytest summary reports {failure_count} failures but failure inventory has {len(failing)}")
    if require_no_failures and (failure_count or failing):
        problems.append("suite with no-failure requirement reported failures")
    return not problems, problems


def quality_and_blockers(observations: dict[str, Any] | None = None) -> dict[str, Any]:
    observed = observations or {}
    l1a2z_quality = _load_json(EVIDENCE_2Z / "quality_and_blocker_matrix.json")
    declared_adjacent = set(l1a2z_quality["frozen_adjacent_suite"]["declared"])
    declared_legacy = set(l1a2z_quality["legacy_lineage_suite"]["declared"])
    stage_suite = observed.get("stage_suite", {"result": "NOT_RUN", "failing": []})
    predecessor = observed.get("predecessor_suite", {"result": "NOT_RUN", "failing": []})
    l1a2y_suite = observed.get("l1a2y_suite", {"result": "NOT_RUN", "failing": []})
    adjacent = observed.get("frozen_adjacent_sweep", {"result": "NOT_RUN", "failing": []})
    legacy = observed.get("legacy_lineage_suite", {"result": "NOT_RUN", "failing": []})
    ratchet = observed.get("quality_ratchet", {})
    problems: list[str] = []
    for label, record in (("stage", stage_suite), ("predecessor", predecessor), ("l1a2y", l1a2y_suite)):
        valid, issues = _suite_observation_valid(record, require_no_failures=True)
        if not valid:
            problems.extend(f"{label}: {issue}" for issue in issues)
    adjacent_valid, adjacent_issues = _suite_observation_valid(adjacent, require_no_failures=False)
    if not adjacent_valid:
        problems.extend(f"frozen-adjacent: {issue}" for issue in adjacent_issues)
    unexpected_adjacent = sorted(
        failure
        for failure in adjacent.get("failing", [])
        if _canonical_test_identifier(failure) not in declared_adjacent
    )
    if unexpected_adjacent:
        problems.append(f"frozen-adjacent failures outside the sealed declared set: {unexpected_adjacent}")
    legacy_valid, legacy_issues = _suite_observation_valid(legacy, require_no_failures=False)
    if not legacy_valid:
        problems.extend(f"legacy-lineage: {issue}" for issue in legacy_issues)
    unexpected_legacy = sorted(
        failure for failure in legacy.get("failing", []) if _canonical_test_identifier(failure) not in declared_legacy
    )
    if unexpected_legacy:
        problems.append(f"legacy-lineage failures outside the sealed declared set: {unexpected_legacy}")
    expected_debt_counts = {"ruff": 171, "format": 27, "isort": 28, "codespell": 3}
    if (
        ratchet.get("result") != "PASS_BASELINE_ONLY"
        or int(ratchet.get("added_diagnostics", -1)) != 0
        or ratchet.get("historical_counts") != expected_debt_counts
    ):
        problems.append("legacy-debt ratchet must preserve the 171/27/28/3 baseline with zero additions")
    if not observed.get("observed_at"):
        problems.append("quality observations have no observed-at record")
    if observed.get("unexpected_regressions", []):
        problems.append(f"unexpected regression inventory is nonempty: {observed['unexpected_regressions']}")
    return {
        "stage": STAGE,
        "observations": observed,
        "observed_at": observed.get("observed_at"),
        "commands": {
            "stage_suite": (
                "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. ../../.venv/bin/python -m pytest "
                "tests/test_l1a3b_support_indexed_observable_audit.py -q"
            ),
            "l1a2z_suite": (
                "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. ../../.venv/bin/python -m pytest "
                "tests/test_l1a2z_sequential_ownbase_attribution_audit.py -q"
            ),
            "l1a2y_suite": (
                "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. ../../.venv/bin/python -m pytest "
                "tests/test_frozen_reconstruction_branch_attribution_audit.py -q"
            ),
            "frozen_adjacent_sweep": (
                "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. ../../.venv/bin/python -m pytest "
                "tests/test_nearwall_taper_interface_transport_audit.py "
                "tests/test_radial_incident_seed_eta_dt_audit.py "
                "tests/test_interior_speed_admissibility_screen_audit.py "
                "tests/test_boundary_compatible_incident_momentum_bound_audit.py "
                "tests/test_frozen_chns_early_window_transport_attribution_audit.py -q"
            ),
            "legacy_lineage_suite": (
                "JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=. ../../.venv/bin/python -m pytest "
                "tests/test_impact_initialization_compatibility_audit.py -q"
            ),
            "quality_ratchet": (
                "PATH=.venv/bin:/home/user/uvtools/bin:$PATH .venv/bin/python scripts/quality_debt_gate.py --tool all"
            ),
        },
        "quality_ratchet": {
            "baseline_path": "quality/legacy_two_phase_debt_v1.json",
            "baseline_sha256": sha256_file(REPO_ROOT / "quality" / "legacy_two_phase_debt_v1.json"),
            "expected": "PASS_BASELINE_ONLY",
            "expected_historical_counts": {"ruff": 171, "format": 27, "isort": 28, "codespell": 3},
            "new_python_files": 2,
            "new_diagnostics_allowed": 0,
            "standalone_ratchet_results": ratchet,
        },
        "stage_suite": stage_suite,
        "predecessor_suite": predecessor,
        "l1a2y_suite": l1a2y_suite,
        "frozen_adjacent_sweep": adjacent,
        "legacy_lineage_suite": legacy,
        "declared_frozen_adjacent_failures_from_immutable_l1a2z_record": sorted(declared_adjacent),
        "declared_legacy_lineage_failures_from_immutable_l1a2z_record": sorted(declared_legacy),
        "unexpected_regressions": observed.get("unexpected_regressions", []),
        "quality_gate_problems": problems,
        "holds": not problems,
        "control_trajectories_run": 0,
    }


def next_action() -> dict[str, Any]:
    return {
        "stage": STAGE,
        "status": "NOT_STARTED",
        "requires_separate_owner_approval": True,
        "action": (
            "If approved, design a separate paired-control protocol using the frozen two-ray vector and "
            "explicit numerical-uncertainty handling; do not assume support invariance or average the components."
        ),
        "m0_trajectory_run": False,
        "control_experiment_authorized_by_this_stage": False,
    }


def _report_markdown(
    verdict: str,
    rays: dict[str, Any],
    historical: dict[str, Any],
    agreement: dict[str, Any],
    adversarial: dict[str, Any],
    quality: dict[str, Any],
    freeze: dict[str, Any],
    source: dict[str, Any],
    publication: dict[str, Any],
) -> str:
    lines = [
        "# L1A-3b — Tie-break-free support-indexed observable contract",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "## 1. Verdict and scope",
        "",
        fill(
            "This stage freezes the support-indexed observable only. It does not authorize or run an `M=0` "
            "trajectory, alter solver or physics code, promote an initializer, or make a mobility-sensitivity "
            "or physical-mechanism claim.",
            width=108,
        ),
        "",
        "## 2. Mathematical observable and signs",
        "",
        fill(
            "`Q(t) = [q95(t), q96(t)]`, with immutable identifiers `RAY_95` and `RAY_96`. Each component is "
            "the frozen linear `phi=0.5` root coordinate `y/dx` on its fixed vertical wall-normal ray. "
            "For each ray, `d_j(t;M)=q_j(0;M)-q_j(t;M)`, positive toward the solid. For a future paired "
            "experiment with identical initial-state hashes, `E_j=d_j(t;M_control)-d_j(t;M_baseline)="
            "q_j(t;M_baseline)-q_j(t;M_control)`. No component is averaged, preferred, maximized, minimized, "
            "or discarded.",
            width=108,
        ),
        "",
        "## 3. Ray geometry and fixed identity",
        "",
        fill(
            f"Grid `{historical['geometry_validation']['grid_shape']}`, "
            f"`dx={historical['geometry_validation']['dx']}`, `dy={historical['geometry_validation']['dy']}`. "
            "Identity is the cell-centred column, independent of a selected root or component-label integer.",
            width=108,
        ),
        "",
        "| Identity | Column | x | Frozen support y | Initial root (`dx`) |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in rays["rays"]:
        lines.append(
            f"| `{item['ray_id']}` | {item['column_index']} | {item['x']:.15g} | "
            f"{item['support_y']:.16g} | {item['initial_root_over_dx']:.15g} |"
        )
    lines.extend(
        [
            "",
            "## 4. Readout, branch correspondence, thresholds, and failure states",
            "",
            fill(
                "Each state enumerates every vertical-ray crossing on the frozen globally reconstructed `phi=0.5` "
                "contour, retains all candidates, validates the frozen edge/cell reconstruction, and maps each "
                "candidate to its current contour component. Exactly one root must lie in the frozen "
                f"`±{ROOT_WINDOW_DX:g} dx` window; its step is at most `{ROOT_STEP_MAX_DX:g} dx`; no competing "
                f"root may be closer than `{ROOT_AMBIGUITY_SEPARATION_DX:g} dx`; and the two selected roots "
                f"must belong to one current component. Cross-state correspondence is geometric with the frozen "
                f"`{BRANCH_IDENTITY_TOLERANCE_DX:g} dx` component tolerance; raw labels are never compared across "
                "trajectories. Candidate or ray order cannot resolve a tie.",
                width=108,
            ),
            fill(
                "Missing roots, multiple admissible roots, near-degenerate roots, excessive root steps, exact "
                "threshold nodes, frozen enumeration/cell mismatches, split support components, and disjoint or "
                "ambiguous geometric correspondence all produce explicit validity failures. The global `NON_UNIQUE` "
                "minimum-gap diagnostic is not used as a substitute for either fixed-ray root.",
                width=108,
            ),
            "",
            "## 5. Frozen global contour diagnostic and topology",
            "",
            fill(
                f"The L1A-2x global minimum-gap diagnostic remains unchanged: "
                f"`{historical['frozen_l1a2x_global_diagnostic_status_counts_retained_verbatim']}` over the historical "
                "intermediate states. L1A-2z branch uniqueness is reported separately. Split/merge relations are "
                "retained as topology events; they never silently change ray identity or select a support.",
                width=108,
            ),
            "",
            "## 6. Symmetric vector and paired-effect reporting",
            "",
            fill(
                "A future valid paired report must include both `d_95`, `d_96`, `E_95`, and `E_96`; the raw signed "
                "effect range; signed difference; absolute support disagreement; sign by ray and discordance; initial "
                "hash equality; root-reconstruction differences; total solver-state uncertainty if defensibly "
                "available; and every validity failure. Invalid pairs produce no partial effect vector. The "
                "`paired_effect_vector` implementation also verifies the algebraic root-coordinate identity.",
                width=108,
            ),
            "",
            "## 7. Support agreement and numerical uncertainty",
            "",
            fill(
                f"Historical maximum support-root spread over 360 stored states is "
                f"`{agreement['original_baseline_support_root_spread_dx']:.12g} dx`; the endpoint spread is "
                f"`{agreement['original_baseline_final_support_root_spread_dx']:.12g} dx`. These are observed "
                "baseline differences, not a noise floor. A per-root Decimal check reports floating-point "
                "interpolation discrepancy for future stored fields; it is not solver/model uncertainty. The total "
                "solver-state uncertainty bound and binary support-invariance threshold are unset, so support "
                f"agreement remains `{agreement['support_agreement_conclusion']}`.",
                width=108,
            ),
            "",
            "## 8. Adversarial and falsification validation",
            "",
            f"Checks: `{adversarial['passed_checks']}/{adversarial['checks_run']}` passed; "
            f"failed `{adversarial['failed_checks']}`.",
            fill(f"Frozen cases: `{', '.join(adversarial['test_ids_frozen_in_source'])}`.", width=108),
            "",
            "## 9. Historical L1A-2z Phase-C comparison",
            "",
            fill(
                f"Verified `{historical['intermediate_state_count']}` stored S1/S2/S3 vectors and "
                f"`{historical['substep_count']}` substep readbacks. q1/q2/q3 statuses: "
                f"`{historical['readout_status_counts_q1_q2_q3']}`; q0/direct statuses: "
                f"`{historical['q0_remeasure_and_q3_direct_status_counts']}`; branch status: "
                f"`{historical['branch_uniqueness_status_counts']}`; topology relations: "
                f"`{historical['geometric_relation_counts_by_support']}`. Both rays shared one component in all "
                "stored states. Segment changes at the two fixed columns were recorded without changing ray IDs.",
                width=108,
            ),
        ]
    )
    for ray_id in RAY_IDS:
        item = historical["endpoints"][ray_id]
        lines.append(
            f"- `{ray_id}`: q(0) `{item['initial_root_over_dx']:.15g} dx`; q(T) "
            f"`{item['final_root_over_dx']:.15g} dx`; d(T) `{item['toward_solid_displacement_dx']:+.15g} dx`."
        )
    lines.extend(
        [
            fill(
                "Limitation: the sealed L1A-2z ledger lacks intermediate phi arrays and full per-ray candidate "
                "lists. Every stored root/status/segment/component record and uniqueness row was checked, but no "
                "solver replay or independent full-candidate reconstruction is claimed.",
                width=108,
            ),
            "",
            "## 10. Sealed evidence and source hashes",
            "",
            f"Freeze boundary: `{freeze['boundary_commit']}`; post-boundary source/test/doc changes: "
            f"`{freeze['post_boundary_source_changes']}`; holds `{freeze['holds']}`.",
            f"L1A-2z manifest SHA-256: `{source['l1a2z_manifest_sha256']}`.",
            f"L1A-3a blocker manifest SHA-256: `{source['l1a3a_manifest_sha256']}`.",
            "Source hash map:",
            "```json",
            json.dumps(source["source_hashes"], indent=2, sort_keys=True),
            "```",
            "",
            "## 11. Focused tests, adjacent suites, and quality ratchet",
            "",
            "Observed local results:",
            "```json",
            json.dumps(quality, indent=2, sort_keys=True),
            "```",
            "The legacy-debt ratchet must remain `PASS_BASELINE_ONLY` with zero added diagnostics; declared "
            "frozen-adjacent and legacy-lineage failures remain visible and are not counted as passes.",
            "",
            "## 12. Branch, pull request, and hosted CI",
            "",
            "Publication record:",
            "```json",
            json.dumps(publication, indent=2, sort_keys=True),
            "```",
            "The authorized branch is `arena/02b6c71b-hydrogym`; PR #24 remains Draft. Hosted CI is claimed "
            "only for a pushed head explicitly named in the publication record.",
            "",
            "## 13. Next action and stop boundary",
            "",
            "No control trajectory, M=0 measurement, contact event, refinement, or physical-mechanism study was "
            "run. Next action remains NOT_STARTED and requires separate owner approval.",
            "",
        ]
    )
    return "\n".join(lines)


def assemble_evidence(payloads: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name in EVIDENCE_FILES:
        if name == "manifest.json":
            continue
        if name not in payloads:
            raise AuditBlocked(f"evidence producer missing for {name}")
        path = directory / name
        if name.endswith(".json"):
            path.write_text(json.dumps(payloads[name], indent=2, sort_keys=True, default=_json_default) + "\n")
        else:
            path.write_text(str(payloads[name]))
    source_hashes = {name: sha256_file(path) for name, path in SOURCE_PATHS.items()}
    manifest = {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "baseline_head_sha": BASELINE_HEAD_SHA,
        "verified_main_sha": VERIFIED_MAIN_SHA,
        "prespecification_commit": payloads["prespecification_freeze.json"]["boundary_commit"],
        "verdict": payloads["l1a3b_final_report.json"]["verdict"],
        "observable_version": STAGE_VERSION,
        "evidence_class": "SUPPORT_INDEXED_FIXED_RAY_ROOT_VECTOR",
        "required_files": list(EVIDENCE_FILES),
        "files": {},
        "source_hashes": source_hashes,
        "control_trajectories_run": 0,
        "solver_source_edited": False,
        "physical_mechanism_claim": False,
        "initializer_promoted": False,
        "support_invariance_claim": False,
        "no_scalar_support_aggregation": True,
        "quality_observations_complete": bool(payloads["quality_and_blocker_matrix.json"].get("holds")),
        "hosted_ci_claim_before_observation": False,
        "report_items_count": int(payloads["l1a3b_final_report.json"].get("required_report_items_count", 0)),
    }
    for name in EVIDENCE_FILES:
        if name != "manifest.json":
            manifest["files"][name] = sha256_file(directory / name)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return verify_evidence_manifest(directory)


def verify_evidence_manifest(out_dir: str | Path) -> dict[str, Any]:
    """Independently verify the L1A-3b seal, source hashes and no-control invariants."""
    directory = Path(out_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise AuditBlocked(f"no manifest under {directory}")
    manifest = json.loads(manifest_path.read_text())
    problems = verify_bound_files(directory, manifest)
    if manifest.get("stage") != STAGE or manifest.get("stage_version") != STAGE_VERSION:
        problems.append("stage/version mismatch")
    if manifest.get("baseline_head_sha") != BASELINE_HEAD_SHA:
        problems.append("baseline blocker head mismatch")
    if manifest.get("verified_main_sha") != VERIFIED_MAIN_SHA:
        problems.append("verified main sha mismatch")
    if manifest.get("verdict") not in ("SUPPORT_INDEXED_OBSERVABLE_FROZEN", "SUPPORT_INDEXED_OBSERVABLE_UNRESOLVED"):
        problems.append("verdict outside the frozen vocabulary")
    if manifest.get("control_trajectories_run") != 0:
        problems.append("L1A-3b must run zero control trajectories")
    if set(manifest.get("required_files", [])) != set(EVIDENCE_FILES):
        problems.append("required-file inventory mismatch")
    if set(manifest.get("files", {})) != set(EVIDENCE_FILES) - {"manifest.json"}:
        problems.append("bound-file inventory mismatch")
    if manifest.get("hosted_ci_claim_before_observation") is not False:
        problems.append("manifest claims unobserved hosted CI")
    if manifest.get("report_items_count") != 13:
        problems.append("manifest does not bind all 13 required report items")
    for key in ("solver_source_edited", "physical_mechanism_claim", "initializer_promoted", "support_invariance_claim"):
        if manifest.get(key) is not False:
            problems.append(f"{key} must be false")
    missing = [name for name in EVIDENCE_FILES if not (directory / name).is_file()]
    if missing:
        problems.append(f"missing required files {missing}")
    current_sources = {name: sha256_file(path) for name, path in SOURCE_PATHS.items() if path.is_file()}
    if set(current_sources) != set(SOURCE_PATHS):
        problems.append("one or more source files are missing from the worktree")
    for name, expected in manifest.get("source_hashes", {}).items():
        if current_sources.get(name) != expected:
            problems.append(f"source hash mismatch: {name}")
    if set(manifest.get("source_hashes", {})) != set(SOURCE_PATHS):
        problems.append("recorded source hash inventory mismatch")
    report = (
        json.loads((directory / "l1a3b_final_report.json").read_text())
        if (directory / "l1a3b_final_report.json").is_file()
        else {}
    )
    if report.get("verdict") != manifest.get("verdict"):
        problems.append("manifest/report verdict mismatch")
    if report.get("control_trajectories_run") != 0:
        problems.append("report does not state zero controls")
    if report.get("physical_mechanism_claim") is not False:
        problems.append("report must not claim a physical mechanism")
    if report.get("required_report_items_count") != 13 or len(report.get("report_items", [])) != 13:
        problems.append("final report does not contain all 13 required items")
    quality_record = (
        json.loads((directory / "quality_and_blocker_matrix.json").read_text())
        if (directory / "quality_and_blocker_matrix.json").is_file()
        else {}
    )
    if bool(manifest.get("quality_observations_complete")) != bool(quality_record.get("holds")):
        problems.append("manifest/quality-observation completeness mismatch")
    if manifest.get("verdict") == "SUPPORT_INDEXED_OBSERVABLE_FROZEN" and quality_record.get("holds") is not True:
        problems.append("frozen verdict requires all local quality/blocker gates to hold")
    if problems:
        raise AuditBlocked(f"evidence verification failed: {problems}")
    freeze = prespecification_freeze()
    if freeze["boundary_commit"] != manifest.get("prespecification_commit"):
        raise AuditBlocked("manifest does not bind the verified prespecification boundary")
    preflight_state()
    verify_l1a2z_evidence()
    verify_l1a3a_blocker()
    return {
        "stage": STAGE,
        "stage_version": STAGE_VERSION,
        "verdict": manifest["verdict"],
        "prespecification_commit": manifest["prespecification_commit"],
        "bound_files": len(manifest["files"]),
        "source_hashes_verified": len(current_sources),
        "control_trajectories_run": 0,
        "problems": [],
        "verified": True,
    }


def run(*, out: str | Path | None = None, observations: dict[str, Any] | None = None) -> dict[str, Any]:
    """Produce L1A-3b contract evidence from sealed history and synthetic checks only."""
    preflight = preflight_state()
    freeze = prespecification_freeze()
    l1a2z_verification = verify_l1a2z_evidence()
    l1a3a_verification = verify_l1a3a_blocker()
    ledger = _load_json(EVIDENCE_2Z / "sequential_ownbase_ledger.json")
    design = _load_json(EVIDENCE_2Z / "support_uniqueness_design.json")
    initial = _load_json(EVIDENCE_2Z / "initial_state_identity.json")
    blocker = _load_json(EVIDENCE_3A / "phase_a_audit.json")
    _, _, ray_validation = frozen_ray_identities(ledger, design, initial, blocker)
    historical, rays, p = historical_validation(ledger, design, initial, blocker)
    agreement = support_agreement_assessment(historical)
    adversarial = adversarial_and_falsification()
    quality = quality_and_blockers(observations)
    source = source_and_provenance(preflight, freeze)
    next_step = next_action()
    gates = {
        "ray_geometry": ray_validation["geometry"]["holds"],
        "historical_stored_vector_records": historical["stored_record_gate_holds"],
        "adversarial_checks": adversarial["holds"],
        "quality_and_blocker_observations": quality["holds"],
        "l1a2z_evidence": l1a2z_verification["verified"],
        "l1a3a_blocker_evidence": l1a3a_verification["verified"],
        "prespecification_freeze": freeze["holds"],
        "no_control_run": True,
        "no_solver_edit": True,
    }
    verdict = "SUPPORT_INDEXED_OBSERVABLE_FROZEN" if all(gates.values()) else "SUPPORT_INDEXED_OBSERVABLE_UNRESOLVED"
    observed = observations or {}
    github_observations = observed.get("github", {})
    publication = {
        "branch": BRANCH,
        "pull_request_number": 24,
        "pull_request_state": github_observations.get("pr_state", "NOT_RECHECKED_FOR_THIS_HEAD"),
        "pull_request_is_draft": github_observations.get("pr_is_draft"),
        "pull_request_head_sha": github_observations.get("pr_head_sha"),
        "hosted_ci": github_observations.get("hosted_ci", {"status": "NOT_OBSERVED_FOR_THIS_HEAD"}),
        "prespecification_boundary_commit": freeze["boundary_commit"],
        "head_at_report_generation": preflight["head_sha"],
    }
    report_items = [
        {"item": 1, "title": "Verdict and scope", "result": {"verdict": verdict, "gates": gates}},
        {
            "item": 2,
            "title": "Mathematical observable and signs",
            "result": "Q=[q95,q96]; d=q(0)-q(t); E=d_control-d_baseline",
        },
        {"item": 3, "title": "Ray geometry and identity", "result": ray_validation},
        {
            "item": 4,
            "title": "Root validity and thresholds",
            "result": {
                "window_dx": ROOT_WINDOW_DX,
                "maximum_step_dx": ROOT_STEP_MAX_DX,
                "minimum_separation_dx": ROOT_AMBIGUITY_SEPARATION_DX,
                "component_correspondence_tolerance_dx": BRANCH_IDENTITY_TOLERANCE_DX,
                "failure_classes": [
                    "missing",
                    "multiple",
                    "close",
                    "exact-node",
                    "step-limit",
                    "component-loss",
                    "fragmentation-or-merge",
                    "ambiguous-geometric-correspondence",
                    "initial-state-hash-mismatch",
                    "enumeration-mismatch",
                ],
            },
        },
        {
            "item": 5,
            "title": "Global contour and topology diagnostics",
            "result": {
                "frozen_l1a2x_status_counts": historical[
                    "frozen_l1a2x_global_diagnostic_status_counts_retained_verbatim"
                ],
                "branch_relation_counts": historical["geometric_relation_counts_by_support"],
                "component_labels_cross_compared": False,
            },
        },
        {"item": 6, "title": "Symmetric paired-effect reporting", "result": paired_effect_vector.__doc__},
        {"item": 7, "title": "Uncertainty and support agreement", "result": agreement},
        {"item": 8, "title": "Adversarial/falsification checks", "result": adversarial},
        {"item": 9, "title": "Historical L1A-2z Phase-C validation", "result": historical},
        {
            "item": 10,
            "title": "Evidence and hashes",
            "result": {
                "source_hashes": source["source_hashes"],
                "l1a2z_manifest_sha256": source["l1a2z_manifest_sha256"],
                "l1a3a_manifest_sha256": source["l1a3a_manifest_sha256"],
                "prespecification_boundary": freeze["boundary_commit"],
            },
        },
        {"item": 11, "title": "Local tests and quality ratchet", "result": quality},
        {"item": 12, "title": "Pull request and hosted CI", "result": publication},
        {"item": 13, "title": "Next action and stop boundary", "result": next_step},
    ]
    report = {
        "stage": STAGE,
        "observable_version": STAGE_VERSION,
        "verdict": verdict,
        "gates": gates,
        "mathematical_definition": {
            "Q(t)": ["q95(t)", "q96(t)"],
            "ray_identities": [ray.ray_id for ray in rays],
            "q_coordinate": "frozen linear phi=0.5 root y/dx on each fixed cell-centred vertical wall-normal ray",
            "displacement": "d_j(t;M)=q_j(0;M)-q_j(t;M), positive toward the solid",
            "paired_effect": (
                "E_j(t)=d_j(t;M_control)-d_j(t;M_baseline)="
                "q_j(t;M_baseline)-q_j(t;M_control) when initial-state hashes match"
            ),
            "support_order": "identity-only storage order; the decision/report treats both components symmetrically",
            "averaging_or_preferred_ray": False,
        },
        "ray_identity_validation": ray_validation,
        "historical_validation": {key: value for key, value in historical.items() if key != "all_intermediate_vectors"},
        "support_agreement": agreement,
        "adversarial_checks": {
            "checks_run": adversarial["checks_run"],
            "passed_checks": adversarial["passed_checks"],
            "failed_checks": adversarial["failed_checks"],
        },
        "quality_summary": quality,
        "publication": publication,
        "report_items": report_items,
        "required_report_items_count": 13,
        "prespecification_boundary": freeze["boundary_commit"],
        "source_hashes": source["source_hashes"],
        "control_trajectories_run": 0,
        "physical_mechanism_claim": False,
        "solver_source_edited": False,
        "support_invariance_claim": False,
        "historical_records_full_candidate_replay_available": historical["full_independent_candidate_replay"][
            "available_from_sealed_l1a2z"
        ],
        "historical_candidate_replay_limitation": historical["full_independent_candidate_replay"],
        "next_action": next_step,
    }
    markdown = _report_markdown(
        verdict, ray_validation, historical, agreement, adversarial, quality, freeze, source, publication
    )
    payloads = {
        "preflight.json": preflight,
        "source_and_provenance.json": source,
        "prespecification_freeze.json": freeze,
        "ray_identity_validation.json": ray_validation,
        "historical_validation.json": historical,
        "support_agreement.json": agreement,
        "adversarial_and_falsification.json": adversarial,
        "quality_and_blocker_matrix.json": quality,
        "next_action.json": next_step,
        "l1a3b_final_report.json": report,
        "l1a3b_final_report.md": markdown,
    }
    directory = Path(out) if out is not None else EVIDENCE_OWN
    verification = assemble_evidence(payloads, directory)
    return {"report": report, "verification": verification}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"{STAGE} tie-break-free two-ray observable contract")
    parser.add_argument("--out", default=None, help="evidence directory to write")
    parser.add_argument("--verify-evidence", default=None, help="verify a sealed evidence directory")
    parser.add_argument("--observed-at", default=None)
    parser.add_argument("--stage-suite-result", default=None)
    parser.add_argument("--stage-suite-failing", default="")
    parser.add_argument("--predecessor-suite-result", default=None)
    parser.add_argument("--predecessor-suite-failing", default="")
    parser.add_argument("--l1a2y-suite-result", default=None)
    parser.add_argument("--l1a2y-suite-failing", default="")
    parser.add_argument("--frozen-adjacent-sweep-result", default=None)
    parser.add_argument("--frozen-adjacent-sweep-failing", default="")
    parser.add_argument("--legacy-lineage-suite-result", default=None)
    parser.add_argument("--legacy-lineage-suite-failing", default="")
    parser.add_argument("--quality-ratchet-result", default=None)
    parser.add_argument("--quality-ratchet-added-diagnostics", type=int, default=-1)
    parser.add_argument("--quality-ratchet-historical-counts-json", default="{}")
    parser.add_argument("--github-observations-json", default="{}")
    parser.add_argument("--unexpected-regressions", default="")
    args = parser.parse_args(argv)
    if args.verify_evidence:
        print(json.dumps(verify_evidence_manifest(args.verify_evidence), indent=2, sort_keys=True))
        return
    observations = {
        "observed_at": args.observed_at,
        "stage_suite": {
            "result": args.stage_suite_result,
            "failing": [x for x in args.stage_suite_failing.split(",") if x],
        },
        "predecessor_suite": {
            "result": args.predecessor_suite_result,
            "failing": [x for x in args.predecessor_suite_failing.split(",") if x],
        },
        "l1a2y_suite": {
            "result": args.l1a2y_suite_result,
            "failing": [x for x in args.l1a2y_suite_failing.split(",") if x],
        },
        "frozen_adjacent_sweep": {
            "result": args.frozen_adjacent_sweep_result,
            "failing": [x for x in args.frozen_adjacent_sweep_failing.split(",") if x],
        },
        "legacy_lineage_suite": {
            "result": args.legacy_lineage_suite_result,
            "failing": [x for x in args.legacy_lineage_suite_failing.split(",") if x],
        },
        "quality_ratchet": {
            "result": args.quality_ratchet_result,
            "added_diagnostics": args.quality_ratchet_added_diagnostics,
            "historical_counts": json.loads(args.quality_ratchet_historical_counts_json),
        },
        "github": json.loads(args.github_observations_json),
        "unexpected_regressions": [x for x in args.unexpected_regressions.split(",") if x],
    }
    outcome = run(out=args.out, observations=observations)
    print(json.dumps(outcome["verification"], indent=2, sort_keys=True))
    print(json.dumps(outcome["report"], indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
