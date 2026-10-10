from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import scripts.quality3b_owner_verifier as verifier

ROOT = Path(__file__).resolve().parents[1]
ATTESTATION = json.loads((ROOT / verifier.ATTESTATION_PATH).read_text())
MANIFEST = json.loads((ROOT / "quality/approved_migrations.json").read_text())
APPROVAL_COMMIT = "6237cbb9614588b976384ca27e6a7c5d05aa3348"


def test_positive_offline_owner_binding():
    verifier.validate_attestation(copy.deepcopy(ATTESTATION))
    verifier.validate_gate_manifest(copy.deepcopy(MANIFEST), APPROVAL_COMMIT)
    verifier.validate_source_hashes(ROOT)


def test_wrong_owner_actor_fails():
    value = copy.deepcopy(ATTESTATION)
    value["github_record"]["actor"] = "not-the-owner"
    with pytest.raises(ValueError, match="github_record_actor"):
        verifier.validate_attestation(value)


def test_fake_owner_association_fails():
    value = copy.deepcopy(ATTESTATION)
    value["github_record"]["author_association"] = "CONTRIBUTOR"
    with pytest.raises(ValueError, match="github_record_owner_association"):
        verifier.validate_attestation(value)


def test_edited_comment_fails():
    value = copy.deepcopy(ATTESTATION)
    value["github_record"]["updated_at"] = "2026-10-10T13:47:00Z"
    with pytest.raises(ValueError, match="github_record_was_edited"):
        verifier.validate_attestation(value)


def test_comment_body_hash_mismatch_fails():
    payload = {
        "id": verifier.COMMENT_ID,
        "user": {"login": verifier.OWNER, "id": 156068376},
        "author_association": "OWNER",
        "created_at": "2026-10-10T13:46:35Z",
        "updated_at": "2026-10-10T13:46:35Z",
        "body": "changed body",
    }
    with pytest.raises(ValueError, match="live_comment_body_hash"):
        verifier.validate_comment_payload(payload, ATTESTATION["github_record"]["body_sha256"])


def test_wrong_approval_commit_fails():
    with pytest.raises(ValueError, match="gate_manifest_approval_commit"):
        verifier.validate_gate_manifest(copy.deepcopy(MANIFEST), "a" * 40)


def test_unowned_approval_commit_fails():
    payload = {
        "sha": APPROVAL_COMMIT,
        "author": {"login": "not-the-owner"},
        "committer": {"login": verifier.OWNER},
        "files": [{"filename": verifier.ATTESTATION_PATH}],
    }
    with pytest.raises(ValueError, match="approval_commit_owner_api_identity"):
        verifier.validate_commit_payload(payload, APPROVAL_COMMIT)


def test_extra_commit_file_fails():
    payload = {
        "sha": APPROVAL_COMMIT,
        "author": {"login": verifier.OWNER},
        "committer": {"login": verifier.OWNER},
        "files": [
            {"filename": verifier.ATTESTATION_PATH},
            {"filename": "examples/two_phase/phasefield.py"},
        ],
    }
    with pytest.raises(ValueError, match="approval_commit_scope"):
        verifier.validate_commit_payload(payload, APPROVAL_COMMIT)


def test_old_sha_one_bit_change_fails():
    value = copy.deepcopy(ATTESTATION)
    value["transitions"][0]["old_sha256"] = "0" + value["transitions"][0]["old_sha256"][1:]
    with pytest.raises(ValueError, match="attestation_transitions_exact_set_mismatch"):
        verifier.validate_attestation(value)


def test_new_sha_one_bit_change_fails():
    value = copy.deepcopy(MANIFEST)
    value["entries"][1]["new_sha256"] = "0" + value["entries"][1]["new_sha256"][1:]
    with pytest.raises(ValueError, match="gate_manifest_exact_set_mismatch"):
        verifier.validate_gate_manifest(value, APPROVAL_COMMIT)


def test_unlisted_fifth_file_fails():
    value = copy.deepcopy(ATTESTATION)
    value["transitions"].append(
        {
            "path": "examples/two_phase/production/not_authorized.py",
            "old_sha256": "1" * 64,
            "new_sha256": "2" * 64,
            "change_class": "LEGACY_SHIFTED_APPROVED",
        }
    )
    with pytest.raises(ValueError, match="attestation_transitions_count"):
        verifier.validate_attestation(value)


def test_wrong_pr22_head_fails():
    value = copy.deepcopy(ATTESTATION)
    value["pr22_head_sha"] = "0" * 40
    with pytest.raises(ValueError, match="attestation_pr22_head"):
        verifier.validate_attestation(value)


def test_scope_expansion_fails():
    value = copy.deepcopy(ATTESTATION)
    value["scope"]["change_frozen_production_sources"] = True
    with pytest.raises(ValueError, match="attestation_scope_not_restrictive"):
        verifier.validate_attestation(value)
