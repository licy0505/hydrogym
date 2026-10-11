#!/usr/bin/env python3
"""Fail-closed verifier for the QUALITY-3B owner attestation.

The migration gate validates manifest shape and exact SHA pairs, but a manifest
field is not proof of GitHub ownership. This verifier independently checks the
live PR comment and the GitHub API metadata for the two-stage attestation
commit before the manifest can be called owner-approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY = "licy0505/hydrogym"
OWNER = "licy0505"
PR_NUMBER = 22
PR22_HEAD = "19da2a4d695e6cceabd0e6c3dcc505d09ba84daf"
PR22_BASE = "002120f3a051e638a7e85f9db4022107dbe45780"
PR23_MERGE = "039f6a0d9a14ff601c76a58f5a66ce0900677de5"
EVIDENCE_HASH = "2187be0a33e2a68d49feb3cb4075883df3bd506e365b6562cd2b161e3aacff5d"
BASELINE_SHA256 = "52b41ab68b4d6b86aaa254389eb5398e65ad8c5cbdcd12626c0466248dd3d70b"
ATTESTATION_PATH = "quality/approvals/quality3b_owner_attestation.json"
COMMENT_ID = 6098141190
COMMENT_API_PATH = f"repos/{REPOSITORY}/issues/comments/{COMMENT_ID}"

TRANSITIONS = (
    {
        "path": "examples/two_phase/production/impact_impulse_projection_audit.py",
        "old_sha256": "b3da6608e06704c56bc7e9ed7f4d0c7d7eed54769635c0cf83c424e182690691",
        "new_sha256": "5a8a4d476e0daa7d9249da1eb0a536011021e68e6ff535703d76488fd122a9b9",
        "change_class": "LEGACY_SHIFTED_APPROVED",
    },
    {
        "path": "examples/two_phase/tests/test_inactive_phase_coupling_audit.py",
        "old_sha256": "869c53b6050aa1ecee9b5da611415fbae15868a658b1746ae604ce5e73f381f7",
        "new_sha256": "1c97db79f830d5387ffede08a4a4f5b4e65139eda68fa407bce8cb2ebf7b5c29",
        "change_class": "LEGACY_SHIFTED_APPROVED",
    },
    {
        "path": "examples/two_phase/tests/test_l1a_data_readiness_exit_audit.py",
        "old_sha256": "ec4e780169391b96f0af67416dab015fb47254893a1c8e05a8c9c3ea655f379c",
        "new_sha256": "e93a05cd5d9a9140e1294f36c788cf54540f8a9476b2eb93c7ac4a8c8083a94f",
        "change_class": "LEGACY_SHIFTED_APPROVED",
    },
    {
        "path": "examples/two_phase/tests/test_stationarity_metric_domain_audit.py",
        "old_sha256": "808f52d749797af44ded20889086d919dc071ec604f0d107797baf936fa60b76",
        "new_sha256": "2418f24262dcb6593586cd2fad4506113ad64878a72360aba8583ba31ae6a215",
        "change_class": "LEGACY_SHIFTED_APPROVED",
    },
)


def _fail(message: str) -> None:
    raise ValueError(f"QUALITY3B_OWNER_VERIFY_FAIL:{message}")


def _full_sha(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"unreadable_json:{path}:{exc}")
    if not isinstance(value, dict):
        _fail(f"json_root_not_object:{path}")
    return value


def _transition_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (item.get("path", ""), item.get("old_sha256", ""), item.get("new_sha256", ""), item.get("change_class", ""))


def expected_transition_keys() -> set[tuple[str, str, str, str]]:
    return {_transition_key(item) for item in TRANSITIONS}


def validate_transition_list(items: Any, label: str = "transitions") -> None:
    if not isinstance(items, list):
        _fail(f"{label}_not_list")
    if len(items) != len(TRANSITIONS):
        _fail(f"{label}_count:{len(items)}")
    for item in items:
        if not isinstance(item, dict):
            _fail(f"{label}_item_not_object")
        if not _full_sha(item.get("old_sha256")) or not _full_sha(item.get("new_sha256")):
            _fail(f"{label}_sha_not_full:{item.get('path')}")
    actual = {_transition_key(item) for item in items}
    if actual != expected_transition_keys():
        _fail(f"{label}_exact_set_mismatch")


def validate_attestation(attestation: dict[str, Any]) -> None:
    if attestation.get("schema") != "quality3b_owner_attestation_v1":
        _fail("attestation_schema")
    if attestation.get("repository") != REPOSITORY or attestation.get("pull_request") != PR_NUMBER:
        _fail("attestation_repo_or_pr")
    if attestation.get("pr22_head_sha") != PR22_HEAD:
        _fail("attestation_pr22_head")
    if attestation.get("pr22_base_sha") != PR22_BASE:
        _fail("attestation_pr22_base")
    if attestation.get("pr23_merge_sha") != PR23_MERGE:
        _fail("attestation_pr23_merge")
    if attestation.get("evidence_hash") != EVIDENCE_HASH:
        _fail("attestation_evidence_hash")
    if attestation.get("baseline_sha256") != BASELINE_SHA256:
        _fail("attestation_baseline_hash")
    validate_transition_list(attestation.get("transitions"), "attestation_transitions")
    scope = attestation.get("scope")
    if not isinstance(scope, dict) or any(value is not False for value in scope.values()):
        _fail("attestation_scope_not_restrictive")
    record = attestation.get("github_record")
    if not isinstance(record, dict):
        _fail("github_record_missing")
    if record.get("kind") != "pull_request_issue_comment":
        _fail("github_record_kind")
    if record.get("comment_id") != COMMENT_ID:
        _fail("github_record_comment_id")
    if record.get("actor") != OWNER or record.get("actor_id") != 156068376:
        _fail("github_record_actor")
    if record.get("author_association") != "OWNER":
        _fail("github_record_owner_association")
    if not _full_sha(record.get("body_sha256")) or not record.get("body_verified_against_api"):
        _fail("github_record_body_proof")
    if record.get("created_at") != record.get("updated_at"):
        _fail("github_record_was_edited")


def validate_gate_manifest(manifest: dict[str, Any], approval_commit: str) -> None:
    if manifest.get("version") != "quality_migrations_v1" or manifest.get("baseline_ref") != "legacy_two_phase_debt_v1":
        _fail("gate_manifest_identity")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != len(TRANSITIONS):
        _fail("gate_manifest_entry_count")
    for entry in entries:
        if not isinstance(entry, dict):
            _fail("gate_manifest_entry_not_object")
        if entry.get("approval_status") != "APPROVED":
            _fail(f"gate_manifest_not_approved:{entry.get('path')}")
        if entry.get("pr22_commit") != PR22_HEAD or entry.get("evidence_hash") != EVIDENCE_HASH:
            _fail(f"gate_manifest_binding:{entry.get('path')}")
        approval = entry.get("approval")
        if not isinstance(approval, dict) or approval.get("approved_by") != OWNER:
            _fail(f"gate_manifest_approved_by:{entry.get('path')}")
        if approval.get("approval_commit") != approval_commit:
            _fail(f"gate_manifest_approval_commit:{entry.get('path')}")
    actual = {
        _transition_key(
            {
                "path": entry.get("path"),
                "old_sha256": entry.get("old_sha256"),
                "new_sha256": entry.get("new_sha256"),
                "change_class": entry.get("change_class"),
            }
        )
        for entry in entries
    }
    if actual != expected_transition_keys():
        _fail("gate_manifest_exact_set_mismatch")


def validate_source_hashes(root: Path) -> None:
    baseline_path = root / "quality" / "legacy_two_phase_debt_v1.json"
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != BASELINE_SHA256:
        _fail("baseline_sha_mismatch")
    for item in TRANSITIONS:
        actual = hashlib.sha256((root / item["path"]).read_bytes()).hexdigest()
        if actual != item["new_sha256"]:
            _fail(f"new_source_sha_mismatch:{item['path']}")


def validate_comment_payload(payload: dict[str, Any], expected_body_sha256: str) -> None:
    if payload.get("id") != COMMENT_ID:
        _fail("live_comment_id")
    if payload.get("user", {}).get("login") != OWNER:
        _fail("live_comment_actor")
    if payload.get("user", {}).get("id") != 156068376:
        _fail("live_comment_actor_id")
    if payload.get("author_association") != "OWNER":
        _fail("live_comment_owner_association")
    if payload.get("updated_at") != payload.get("created_at"):
        _fail("live_comment_edited")
    if hashlib.sha256(str(payload.get("body", "")).encode("utf-8")).hexdigest() != expected_body_sha256:
        _fail("live_comment_body_hash")


def validate_commit_payload(payload: dict[str, Any], approval_commit: str) -> None:
    if payload.get("sha") != approval_commit:
        _fail("approval_commit_sha")
    if payload.get("author", {}).get("login") != OWNER or payload.get("committer", {}).get("login") != OWNER:
        _fail("approval_commit_owner_api_identity")
    files = payload.get("files") or []
    paths = [item.get("filename") for item in files]
    if paths != [ATTESTATION_PATH]:
        _fail(f"approval_commit_scope:{paths}")


def _gh_api(path: str) -> dict[str, Any]:
    proc = subprocess.run(["gh", "api", path], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        _fail(f"github_api_error:{path}:{proc.stderr.strip()}")
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        _fail(f"github_api_json:{path}:{exc}")
    if not isinstance(result, dict):
        _fail(f"github_api_root:{path}")
    return result


def verify(
    root: Path, attestation_path: Path, manifest_path: Path, approval_commit: str, live: bool = True
) -> dict[str, Any]:
    if not _full_sha(approval_commit, 40):
        _fail("approval_commit_not_full_sha")
    attestation = _load(attestation_path)
    manifest = _load(manifest_path)
    validate_attestation(attestation)
    validate_gate_manifest(manifest, approval_commit)
    validate_source_hashes(root)
    record = attestation["github_record"]
    result: dict[str, Any] = {
        "owner_approval_verified": False,
        "approval_commit": approval_commit,
        "attestation_path": str(attestation_path),
        "gate_manifest_path": str(manifest_path),
        "source_hashes_verified": True,
    }
    if live:
        comment = _gh_api(COMMENT_API_PATH)
        commit = _gh_api(f"repos/{REPOSITORY}/commits/{approval_commit}")
        validate_comment_payload(comment, record["body_sha256"])
        validate_commit_payload(commit, approval_commit)
        result["github_comment"] = {
            "id": comment["id"],
            "url": comment.get("html_url"),
            "actor": comment["user"]["login"],
            "author_association": comment["author_association"],
            "created_at": comment["created_at"],
            "updated_at": comment["updated_at"],
            "body_sha256": record["body_sha256"],
        }
        result["github_commit"] = {
            "sha": commit["sha"],
            "url": commit.get("html_url"),
            "author": commit.get("author", {}).get("login"),
            "committer": commit.get("committer", {}).get("login"),
            "files": [item.get("filename") for item in commit.get("files", [])],
            "signature_verified": bool((commit.get("commit") or {}).get("verification", {}).get("verified")),
        }
    result["owner_approval_verified"] = True
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--attestation", default=ATTESTATION_PATH)
    parser.add_argument("--manifest", default="quality/approved_migrations.json")
    parser.add_argument("--approval-commit", required=True)
    parser.add_argument(
        "--no-live-github", action="store_true", help="Only for isolated unit tests; never use for final evidence."
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    try:
        result = verify(
            root,
            root / args.attestation,
            root / args.manifest,
            args.approval_commit,
            live=not args.no_live_github,
        )
    except ValueError as exc:
        print(str(exc))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
