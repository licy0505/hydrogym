"""Adversarial tests for the Option-R quality-debt ratchet. No CFD work."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

import scripts.quality_debt_gate as gate

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "quality" / "legacy_two_phase_debt_v1.json"


@pytest.fixture(scope="module")
def baseline() -> dict:
    return gate.load_baseline(ROOT, BASELINE)


def test_baseline_schema_and_summary(baseline):
    assert baseline["version"] == "legacy_two_phase_debt_v1"
    assert baseline["summary"]["ruff_findings"] == 171
    assert baseline["summary"]["format_files"] == 27
    assert baseline["summary"]["isort_files"] == 28
    assert baseline["summary"]["codespell_findings"] == 3
    assert len(baseline["ruff"]) == 171
    assert len(baseline["file_sha256"]) >= 12


def test_pinned_phasefield_and_chns_hashes(baseline):
    pf = "examples/two_phase/phasefield.py"
    chns = "examples/two_phase/production/chns_nonstationarity_audit.py"
    assert gate.sha256_file(ROOT / pf) == baseline["file_sha256"][pf]
    assert gate.sha256_file(ROOT / chns) == baseline["file_sha256"][chns]


def test_compare_multiset_equal_count_swap_fails():
    expected = Counter({("a.py", "E501", 1, 1, "line"): 1})
    actual = Counter({("a.py", "E501", 2, 1, "other"): 1})
    errors = gate.compare_multiset(actual, expected, "ruff")
    assert any(e.startswith("NEW_FINDING") for e in errors)
    assert any(e.startswith("MISSING_HISTORICAL") for e in errors)


def test_load_baseline_rejects_extra_exemption(tmp_path, baseline):
    data = json.loads(json.dumps(baseline))
    data["ruff"].append(
        {
            "path": "hydrogym/new.py",
            "code": "E501",
            "row": 1,
            "col": 121,
            "message": "Line too long",
        }
    )
    data["summary"]["ruff_findings"] = 171
    path = tmp_path / "legacy_two_phase_debt_v1.json"
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="ruff record count mismatch"):
        gate.load_baseline(ROOT, path)


def test_load_baseline_rejects_summary_inflation(tmp_path, baseline):
    data = json.loads(json.dumps(baseline))
    data["summary"]["ruff_findings"] = 172
    path = tmp_path / "legacy_two_phase_debt_v1.json"
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="ruff_findings"):
        gate.load_baseline(ROOT, path)


def test_parse_ruff_rejects_invalid_json():
    with pytest.raises(SystemExit, match="unparsable"):
        gate.parse_ruff("{", ROOT)


def test_skipped_is_not_pass():
    assert gate.emit_result("ruff", 1, 171, 0, True, []) == 0
    assert gate.emit_result("ruff", 1, 171, 1, False, ["NEW_FINDING"]) == 1


def test_new_file_e501_is_new_finding(baseline):
    actual = Counter(gate.ruff_key(x) for x in baseline["ruff"])
    extra_item = {
        "path": "scripts/quality_debt_gate.py",
        "code": "E501",
        "row": 1,
        "col": 121,
        "message": "Line too long (200 > 120)",
    }
    actual[gate.ruff_key(extra_item)] += 1
    expected = Counter(gate.ruff_key(x) for x in baseline["ruff"])
    errors = gate.compare_multiset(actual, expected, "ruff")
    assert any("NEW_FINDING" in e for e in errors)


def test_new_file_f841_is_new_finding(baseline):
    extra = {
        "path": "scripts/quality_debt_gate.py",
        "code": "F841",
        "row": 3,
        "col": 5,
        "message": "Local variable `x` is assigned to but never used",
    }
    actual = Counter(gate.ruff_key(x) for x in [*baseline["ruff"], extra])
    expected = Counter(gate.ruff_key(x) for x in baseline["ruff"])
    errors = gate.compare_multiset(actual, expected, "ruff")
    assert any("F841" in e and "NEW_FINDING" in e for e in errors)


def test_whitespace_sha_drift_on_pinned_file(tmp_path, baseline):
    rel = "examples/two_phase/phasefield.py"
    dest = tmp_path / rel
    dest.parent.mkdir(parents=True)
    dest.write_text((ROOT / rel).read_text() + "\n")
    fake = {"file_sha256": {rel: baseline["file_sha256"][rel]}}
    errors = gate.check_pinned_shas(tmp_path, fake, {rel})
    assert any(e.startswith("SHA_DRIFT") for e in errors)


def test_missing_pinned_file_fails(tmp_path, baseline):
    rel = "examples/two_phase/phasefield.py"
    fake = {"file_sha256": {rel: baseline["file_sha256"][rel]}}
    errors = gate.check_pinned_shas(tmp_path, fake, set())
    assert any(e.startswith("MISSING_PINNED_FILE") for e in errors)


def test_format_new_file_fails(baseline):
    actual = Counter([*baseline["format_files"], "scripts/quality_debt_gate.py"])
    expected = Counter(baseline["format_files"])
    errors = gate.compare_multiset(actual, expected, "format")
    assert any("NEW_FINDING:format" in e for e in errors)


def test_isort_new_file_fails(baseline):
    actual = Counter([*baseline["isort_files"], "scripts/quality_debt_gate.py"])
    expected = Counter(baseline["isort_files"])
    errors = gate.compare_multiset(actual, expected, "isort")
    assert any("NEW_FINDING:isort" in e for e in errors)


def test_codespell_new_token_fails(baseline):
    extra = {"path": "README.md", "row": 1, "token": "not-a-legacy-token"}
    actual = Counter([gate.codespell_key(x) for x in baseline["codespell"]] + [gate.codespell_key(extra)])
    expected = Counter(gate.codespell_key(x) for x in baseline["codespell"])
    errors = gate.compare_multiset(actual, expected, "codespell")
    assert any("NEW_FINDING:codespell" in e for e in errors)


def test_gate_does_not_claim_security_audit():
    src = Path(gate.__file__).read_text()
    assert "uv audit" not in src
    assert "BASELINE_ONLY_ACCEPTED" in src
    assert "ruff clean" not in src.lower()


@pytest.mark.parametrize("tool", ["ruff", "format", "isort", "codespell"])
def test_live_gate_on_repository(tool):
    code = gate.main(["--tool", tool, "--root", str(ROOT)])
    assert code == 0
