"""Adversarial tests for the Option-R quality-debt ratchet. No CFD work."""

from __future__ import annotations

import json
import subprocess
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
def test_live_gate_on_repository(tool, capsys):
    code = gate.main(["--tool", tool, "--root", str(ROOT)])
    baseline = gate.load_baseline(ROOT, BASELINE)
    overlap = {
        "examples/two_phase/production/impact_impulse_projection_audit.py",
        "examples/two_phase/tests/test_inactive_phase_coupling_audit.py",
        "examples/two_phase/tests/test_l1a_data_readiness_exit_audit.py",
        "examples/two_phase/tests/test_stationarity_metric_domain_audit.py",
    }
    drift = [rel for rel in overlap if gate.sha256_file(ROOT / rel) != baseline["file_sha256"][rel]]
    if drift:
        assert code == 1
        out = capsys.readouterr().out
        assert "SHA_DRIFT:" in out or "PROTECTED_SHA_DRIFT_UNAPPROVED:" in out
    else:
        assert code == 0
        assert "PASS_BASELINE_ONLY" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# QUALITY-3 adversarial coverage (spec section 7, items 1-12).
# Synthetic repositories with real failure injection through gate.main(); no
# CFD work. uv_run is stubbed so each tool's raw output is fully controlled.
# ---------------------------------------------------------------------------


def _ruff_rec(path, code, row, col, message, line_text=None):
    rec = {"path": path, "code": code, "row": row, "col": col, "message": message}
    if line_text is not None:
        rec["line_text"] = line_text
    return rec


def _cs_rec(path, row, token, suggestion="fix"):
    return {"path": path, "row": row, "token": token, "suggestion": suggestion}


def _migration_entry(
    root,
    rel,
    new_content_sha,
    change_class,
    approval_status,
    old_diag=None,
    mapped_diag=None,
    new_diag=None,
    approval=None,
    old_sha_override=None,
    new_sha_override=None,
):
    baseline = json.loads((root / "quality" / "legacy_two_phase_debt_v1.json").read_text())
    entry = {
        "path": rel,
        "old_sha256": old_sha_override or baseline["file_sha256"][rel],
        "new_sha256": new_sha_override or new_content_sha,
        "pr22_commit": "a" * 40,
        "change_class": change_class,
        "old_diagnostics": old_diag or {},
        "mapped_diagnostics": mapped_diag or {},
        "new_diagnostics": new_diag or {},
        "approval_status": approval_status,
        "evidence_hash": "e" * 64,
    }
    if approval is not None:
        entry["approval"] = approval
    return entry


def _write_migrations(root, entries):
    payload = {
        "version": "quality_migrations_v1",
        "baseline_ref": "legacy_two_phase_debt_v1",
        "entries": entries,
    }
    (root / "quality" / "approved_migrations.json").write_text(json.dumps(payload, indent=2))


def _q3_world(
    tmp_path,
    monkeypatch,
    *,
    baseline_ruff=None,
    baseline_format=None,
    baseline_isort=None,
    baseline_codespell=None,
    pinned=None,
    actual_ruff=None,
    actual_format=None,
    actual_isort=None,
    actual_codespell=None,
    faults=None,
):
    """Build a synthetic repo + baseline and stub gate.uv_run deterministically."""
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
    (root / "quality").mkdir(exist_ok=True)
    pinned = pinned or {}
    for rel, content in pinned.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    baseline_ruff = baseline_ruff or []
    baseline_format = baseline_format or []
    baseline_isort = baseline_isort or []
    baseline_codespell = baseline_codespell or []
    baseline = {
        "version": "legacy_two_phase_debt_v1",
        "policy": "option_R_non_increasing_exact_multiset",
        "summary": {
            "ruff_findings": len(baseline_ruff),
            "format_files": len(baseline_format),
            "isort_files": len(baseline_isort),
            "codespell_findings": len(baseline_codespell),
        },
        "file_sha256": {rel: gate.sha256_file(root / rel) for rel in pinned},
        "ruff": baseline_ruff,
        "format_files": baseline_format,
        "isort_files": baseline_isort,
        "codespell": baseline_codespell,
    }
    baseline_path = root / "quality" / "legacy_two_phase_debt_v1.json"
    baseline_path.write_text(json.dumps(baseline))
    monkeypatch.setattr(gate, "REQUIRED_SUMMARY", dict(baseline["summary"]))
    monkeypatch.setattr(gate, "BASELINE_SHA256", gate.sha256_file(baseline_path))

    actual_ruff = actual_ruff if actual_ruff is not None else baseline_ruff
    actual_format = actual_format if actual_format is not None else baseline_format
    actual_isort = actual_isort if actual_isort is not None else baseline_isort
    actual_codespell = actual_codespell if actual_codespell is not None else baseline_codespell
    faults = faults or {}

    def fake_uv_run(_root, rest):
        def faulted(default_rc, default_out, default_err=""):
            fault = faults.get(rest[0])
            if fault is not None:
                return subprocess.CompletedProcess(
                    rest,
                    fault.get("rc", default_rc),
                    fault.get("stdout", default_out),
                    fault.get("stderr", default_err),
                )
            return subprocess.CompletedProcess(rest, default_rc, default_out, default_err)

        if rest[0] == "ruff" and rest[1] == "check":
            payload = json.dumps(
                [
                    {
                        "filename": str(root / rec["path"]),
                        "code": rec["code"],
                        "location": {"row": rec["row"], "column": rec["col"]},
                        "message": rec["message"],
                    }
                    for rec in actual_ruff
                ]
            )
            return faulted(1 if actual_ruff else 0, payload)
        if rest[0] == "ruff" and rest[1] == "format":
            lines = []
            for rel in actual_format:
                lines.append(f"--- {root / rel}\t2026-01-01 00:00:00.000000")
                lines.append(f"+++ {root / rel}\t2026-01-01 00:00:00.000000")
                lines.append("@@ -1 +1 @@")
                lines.append("-old")
                lines.append("+new")
            return faulted(1 if actual_format else 0, "\n".join(lines))
        if rest[0] == "isort":
            err = "".join(
                f"ERROR: {root / rel} Imports are incorrectly sorted and/or formatted.\n" for rel in actual_isort
            )
            return faulted(1 if actual_isort else 0, "", err)
        if rest[0] == "codespell":
            out = "".join(
                f"{rec['path']}:{rec['row']}: {rec['token']} ==> {rec['suggestion']}\n" for rec in actual_codespell
            )
            return faulted(65 if actual_codespell else 0, out)
        raise AssertionError(f"unexpected tool invocation: {rest}")

    monkeypatch.setattr(gate, "uv_run", fake_uv_run)
    return root


def test_q3_01_new_file_findings_get_no_legacy_exemption(tmp_path, monkeypatch, capsys):
    legacy = _ruff_rec("legacy.py", "E501", 10, 121, "Line too long (130 > 120)")
    root = _q3_world(
        tmp_path,
        monkeypatch,
        baseline_ruff=[legacy],
        pinned={"legacy.py": b"x = 1\n"},
        actual_ruff=[
            legacy,
            _ruff_rec("newdir/new.py", "E501", 3, 121, "Line too long (150 > 120)"),
            _ruff_rec("newdir/new.py", "F841", 5, 5, "Local variable `y` is assigned to but never used"),
        ],
    )
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "NEW_FINDING:ruff" in out
    assert "NEW_PATH_NOT_IN_BASELINE:newdir/new.py" in out


def test_q3_02_pinned_sha_drift_fails_with_identical_findings(tmp_path, monkeypatch, capsys):
    legacy = _ruff_rec("legacy.py", "E501", 10, 121, "Line too long (130 > 120)")
    root = _q3_world(
        tmp_path,
        monkeypatch,
        baseline_ruff=[legacy],
        pinned={"legacy.py": b"x = 1\n"},
        actual_ruff=[legacy],
    )
    (root / "legacy.py").write_bytes(b"x = 1  \n")  # approved-looking no-op edit; still drift
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    assert "SHA_DRIFT:legacy.py" in capsys.readouterr().out


def test_q3_03_one_removed_one_added_equal_counts_fails(tmp_path, monkeypatch, capsys):
    old = _ruff_rec("legacy.py", "E501", 10, 121, "Line too long (130 > 120)")
    swapped = _ruff_rec("legacy.py", "E501", 999, 121, "Line too long (131 > 120)")
    root = _q3_world(
        tmp_path,
        monkeypatch,
        baseline_ruff=[old],
        pinned={"legacy.py": b"x = 1\n"},
        actual_ruff=[swapped],
    )
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "NEW_FINDING" in out and "MISSING_HISTORICAL" in out


def test_q3_04_row_shift_needs_diff_evidence_and_owner_approval(tmp_path, monkeypatch, capsys):
    text = "value = 'a very long legacy line that exceeds the limit'\n"
    old = _ruff_rec("legacy.py", "E501", 10, 121, "Line too long (130 > 120)", line_text=text)
    shifted = _ruff_rec("legacy.py", "E501", 42, 121, "Line too long (130 > 120)", line_text=text)
    root = _q3_world(
        tmp_path,
        monkeypatch,
        baseline_ruff=[old],
        pinned={"legacy.py": b"x = 1\n"},
        actual_ruff=[shifted],
    )
    (root / "legacy.py").write_bytes(b"# inserted audit block\nx = 1\n")
    new_sha = gate.sha256_file(root / "legacy.py")
    # A: no manifest -> shift is not auto-inherited
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "NEW_FINDING" in out and "MISSING_HISTORICAL" in out
    diag_old = {"ruff": [old], "format": [], "isort": [], "codespell": []}
    diag_mapped = {"ruff": [shifted], "format": [], "isort": [], "codespell": []}
    diag_empty = {"ruff": [], "format": [], "isort": [], "codespell": []}
    entry_kwargs = dict(
        old_diag=diag_old,
        mapped_diag=diag_mapped,
        new_diag=diag_empty,
    )
    # B: PENDING_OWNER manifest -> still fails (fail closed)
    _write_migrations(
        root,
        [_migration_entry(root, "legacy.py", new_sha, "LEGACY_SHIFTED_APPROVED", "PENDING_OWNER", **entry_kwargs)],
    )
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    assert "PROTECTED_SHA_DRIFT_UNAPPROVED:legacy.py" in capsys.readouterr().out
    # C: APPROVED with exact fingerprint -> passes
    _write_migrations(
        root,
        [
            _migration_entry(
                root,
                "legacy.py",
                new_sha,
                "LEGACY_SHIFTED_APPROVED",
                "APPROVED",
                approval={"approved_by": "owner", "approval_commit": "b" * 40},
                **entry_kwargs,
            )
        ],
    )
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 0
    assert "moved_approved=1" in capsys.readouterr().out
    # D: APPROVED but the code-text fingerprint differs -> rejected at load
    forged = _ruff_rec("legacy.py", "E501", 42, 121, "Line too long (130 > 120)", line_text="different text\n")
    forged_diag = {"ruff": [forged], "format": [], "isort": [], "codespell": []}
    _write_migrations(
        root,
        [
            _migration_entry(
                root,
                "legacy.py",
                new_sha,
                "LEGACY_SHIFTED_APPROVED",
                "APPROVED",
                approval={"approved_by": "owner", "approval_commit": "b" * 40},
                old_diag=diag_old,
                mapped_diag=forged_diag,
                new_diag=diag_empty,
            )
        ],
    )
    with pytest.raises(SystemExit, match="fingerprint mismatch"):
        gate.main(["--tool", "ruff", "--root", str(root)])


def test_q3_05_unchanged_legacy_keeps_passing(tmp_path, monkeypatch, capsys):
    legacy = _ruff_rec("legacy.py", "E501", 10, 121, "Line too long (130 > 120)")
    root = _q3_world(
        tmp_path,
        monkeypatch,
        baseline_ruff=[legacy],
        pinned={"legacy.py": b"x = 1\n"},
        actual_ruff=[legacy],
    )
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "PASS_BASELINE_ONLY" in out and "kept=1" in out


def test_q3_06_approved_removal_then_reintroduction_fails(tmp_path, monkeypatch, capsys):
    keep = _ruff_rec("legacy.py", "E501", 10, 121, "Line too long (130 > 120)")
    gone = _ruff_rec("legacy.py", "E501", 20, 121, "Line too long (144 > 120)")
    root = _q3_world(
        tmp_path,
        monkeypatch,
        baseline_ruff=[keep, gone],
        pinned={"legacy.py": b"x = 1\ny = 2\n"},
        actual_ruff=[keep],
    )
    (root / "legacy.py").write_bytes(b"x = 1\n")  # owner-approved fix removed row 20
    new_sha = gate.sha256_file(root / "legacy.py")
    diag_old = {"ruff": [gone], "format": [], "isort": [], "codespell": []}
    diag_empty = {"ruff": [], "format": [], "isort": [], "codespell": []}
    _write_migrations(
        root,
        [
            _migration_entry(
                root,
                "legacy.py",
                new_sha,
                "LEGACY_REMOVED_APPROVED",
                "APPROVED",
                approval={"approved_by": "owner", "approval_commit": "b" * 40},
                old_diag=diag_old,
                mapped_diag=diag_empty,
                new_diag=diag_empty,
            )
        ],
    )
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 0
    assert "removed_approved=1" in capsys.readouterr().out
    # Reintroducing the removed finding must fail again.
    monkeypatch.setattr(gate, "uv_run", gate.uv_run)  # keep stub; rebuild actual via new world is overkill
    root2 = _q3_world(
        tmp_path / "reintro",
        monkeypatch,
        baseline_ruff=[keep, gone],
        pinned={"legacy.py": b"x = 1\ny = 2\n"},
        actual_ruff=[keep, gone],
    )
    (root2 / "legacy.py").write_bytes(b"x = 1\n")
    _write_migrations(
        root2,
        [
            _migration_entry(
                root2,
                "legacy.py",
                gate.sha256_file(root2 / "legacy.py"),
                "LEGACY_REMOVED_APPROVED",
                "APPROVED",
                approval={"approved_by": "owner", "approval_commit": "b" * 40},
                old_diag=diag_old,
                mapped_diag=diag_empty,
                new_diag=diag_empty,
            )
        ],
    )
    # baseline pin is b"x = 1\n" but content is b"x = 1\n" -> no drift; the removed
    # finding reappears in actual output -> NEW_FINDING against adjusted expected.
    assert gate.main(["--tool", "ruff", "--root", str(root2)]) == 1
    assert "NEW_FINDING" in capsys.readouterr().out


def test_q3_07_baseline_json_edit_cannot_bypass_manifest(tmp_path, monkeypatch):
    legacy = _ruff_rec("legacy.py", "E501", 10, 121, "Line too long (130 > 120)")
    root = _q3_world(
        tmp_path,
        monkeypatch,
        baseline_ruff=[legacy],
        pinned={"legacy.py": b"x = 1\n"},
        actual_ruff=[legacy],
    )
    (root / "legacy.py").write_bytes(b"x = 2\n")
    baseline_path = root / "quality" / "legacy_two_phase_debt_v1.json"
    data = json.loads(baseline_path.read_text())
    data["file_sha256"]["legacy.py"] = gate.sha256_file(root / "legacy.py")  # silent re-pin attempt
    baseline_path.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        gate.main(["--tool", "ruff", "--root", str(root)])
    data["summary"]["ruff_findings"] = 0  # summary edit attempt
    data["ruff"] = []
    baseline_path.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="ruff_findings"):
        gate.main(["--tool", "ruff", "--root", str(root)])


def test_q3_08_new_findings_in_any_directory_fail(tmp_path, monkeypatch, capsys):
    root = _q3_world(
        tmp_path,
        monkeypatch,
        baseline_ruff=[],
        pinned={"sentinel.py": b"# baseline sentinel\n"},
        actual_ruff=[
            _ruff_rec("scripts/tool.py", "F841", 2, 5, "Local variable `z` is assigned to but never used"),
            _ruff_rec("docs/example.py", "E501", 7, 121, "Line too long (125 > 120)"),
            _ruff_rec("hydrogym/core.py", "E501", 9, 121, "Line too long (126 > 120)"),
        ],
    )
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    out = capsys.readouterr().out
    for path in ("scripts/tool.py", "docs/example.py", "hydrogym/core.py"):
        assert f"NEW_PATH_NOT_IN_BASELINE:{path}" in out


def test_q3_09_injected_typo_and_import_disorder_fail(tmp_path, monkeypatch, capsys):
    root = _q3_world(
        tmp_path,
        monkeypatch,
        baseline_codespell=[_cs_rec("legacy.md", 3, "re" + "tuned")],
        baseline_isort=["legacy.py"],
        pinned={"legacy.py": b"import os\nimport sys\n"},
        actual_codespell=[
            _cs_rec("legacy.md", 3, "re" + "tuned"),
            _cs_rec("docs/new.md", 12, "t" + "eh"),
        ],
        actual_isort=["legacy.py", "scripts/new_tool.py"],
    )
    (root / "legacy.md").write_bytes(("re" + "tuned\n").encode())
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "new_tool.py").write_bytes(b"import sys\nimport os\n")
    assert gate.main(["--tool", "codespell", "--root", str(root)]) == 1
    assert "NEW_FINDING:codespell" in capsys.readouterr().out
    assert gate.main(["--tool", "isort", "--root", str(root)]) == 1
    assert "NEW_FINDING:isort" in capsys.readouterr().out


def test_q3_10_tool_failure_bad_output_missing_stage_fail(tmp_path, monkeypatch, capsys):
    legacy = _ruff_rec("legacy.py", "E501", 10, 121, "Line too long (130 > 120)")
    base_kwargs = dict(baseline_ruff=[legacy], pinned={"legacy.py": b"x = 1\n"}, actual_ruff=[legacy])
    # (a) tool cannot start
    root = _q3_world(tmp_path / "a", monkeypatch, faults={"ruff": {"rc": 127, "stdout": ""}}, **base_kwargs)
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    assert "TOOL_CRASH" in capsys.readouterr().out
    # (b) malformed tool output
    root = _q3_world(tmp_path / "b", monkeypatch, faults={"ruff": {"rc": 1, "stdout": "{"}}, **base_kwargs)
    with pytest.raises(SystemExit, match="un" + "parsable"):
        gate.main(["--tool", "ruff", "--root", str(root)])
    # (c) pinned file missing from the scan tree
    root = _q3_world(tmp_path / "c", monkeypatch, **base_kwargs)
    (root / "legacy.py").unlink()
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    assert "MISSING_PINNED_FILE:legacy.py" in capsys.readouterr().out
    # (d) MISSING_STAGE: tool ran but produced no findings the baseline requires
    root = _q3_world(
        tmp_path / "d",
        monkeypatch,
        baseline_ruff=[legacy],
        pinned={"legacy.py": b"x = 1\n"},
        actual_ruff=[],
    )
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    assert "MISSING_HISTORICAL" in capsys.readouterr().out


def test_q3_11_overlap_file_exception_only_under_approved_pair(tmp_path, monkeypatch, capsys):
    root = _q3_world(
        tmp_path,
        monkeypatch,
        baseline_ruff=[],
        pinned={"overlap.py": b"original = True\n"},
        actual_ruff=[],
    )
    (root / "overlap.py").write_bytes("original = True\ndiagnostic_capture = True\n".encode())
    new_sha = gate.sha256_file(root / "overlap.py")
    # (a) no manifest
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    assert "SHA_DRIFT:overlap.py" in capsys.readouterr().out
    # (b) pending manifest -> protected drift unapproved
    _write_migrations(root, [_migration_entry(root, "overlap.py", new_sha, "LEGACY_UNCHANGED", "PENDING_OWNER")])
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    assert "PROTECTED_SHA_DRIFT_UNAPPROVED:overlap.py" in capsys.readouterr().out
    # (c) approved exact pair -> pass with logged transition
    _write_migrations(
        root,
        [
            _migration_entry(
                root,
                "overlap.py",
                new_sha,
                "LEGACY_UNCHANGED",
                "APPROVED",
                approval={"approved_by": "owner", "approval_commit": "b" * 40},
            )
        ],
    )
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 0
    assert "sha_transitions_approved=1" in capsys.readouterr().out
    # (d) approved but paired to a different future hash -> still fails
    _write_migrations(
        root,
        [
            _migration_entry(
                root,
                "overlap.py",
                new_sha,
                "LEGACY_UNCHANGED",
                "APPROVED",
                approval={"approved_by": "owner", "approval_commit": "b" * 40},
                new_sha_override="c" * 64,
            )
        ],
    )
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    assert "PROTECTED_SHA_DRIFT_UNAPPROVED:overlap.py" in capsys.readouterr().out
    # (e) prefix/wildcard sha in the manifest -> rejected at load
    _write_migrations(
        root,
        [
            _migration_entry(
                root,
                "overlap.py",
                new_sha,
                "LEGACY_UNCHANGED",
                "APPROVED",
                approval={"approved_by": "owner", "approval_commit": "b" * 40},
                new_sha_override=new_sha[:16],
            )
        ],
    )
    with pytest.raises(SystemExit, match="64-hex"):
        gate.main(["--tool", "ruff", "--root", str(root)])
    # (f) approved without an approval record -> rejected at load
    _write_migrations(root, [_migration_entry(root, "overlap.py", new_sha, "LEGACY_UNCHANGED", "APPROVED")])
    with pytest.raises(SystemExit, match="approved_by"):
        gate.main(["--tool", "ruff", "--root", str(root)])


def test_q3_12_frozen_physics_sources_always_fail(tmp_path, monkeypatch, capsys):
    pf_rel = "examples/two_phase/phasefield.py"
    gd_rel = "examples/two_phase/generate_dataset.py"
    root = _q3_world(
        tmp_path,
        monkeypatch,
        baseline_ruff=[],
        pinned={pf_rel: b"SOLVER_CONTRACT_VERSION = 12\n", gd_rel: b"# frozen\n"},
        actual_ruff=[],
    )
    (root / pf_rel).write_bytes(b"SOLVER_CONTRACT_VERSION = 13\n")
    # (a) plain drift fails
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    assert f"SHA_DRIFT:{pf_rel}" in capsys.readouterr().out
    # (b) even an APPROVED migration entry is forbidden for frozen physics sources
    _write_migrations(
        root,
        [
            _migration_entry(
                root,
                pf_rel,
                gate.sha256_file(root / pf_rel),
                "LEGACY_UNCHANGED",
                "APPROVED",
                approval={"approved_by": "owner", "approval_commit": "b" * 40},
            )
        ],
    )
    with pytest.raises(SystemExit, match="FROZEN_SOURCE_MIGRATION_FORBIDDEN"):
        gate.main(["--tool", "ruff", "--root", str(root)])
    # (c) generate_dataset.py is equally protected
    (root / pf_rel).write_bytes(b"SOLVER_CONTRACT_VERSION = 12\n")
    (root / gd_rel).write_bytes(b"# frozen but edited\n")
    (root / "quality" / "approved_migrations.json").unlink()
    assert gate.main(["--tool", "ruff", "--root", str(root)]) == 1
    assert f"SHA_DRIFT:{gd_rel}" in capsys.readouterr().out
    # (d) the repository's real migration manifest must never target frozen sources
    real_manifest = ROOT / "quality" / "approved_migrations.json"
    if real_manifest.is_file():
        entries = json.loads(real_manifest.read_text())["entries"]
        for entry in entries:
            assert entry["path"] not in gate.FROZEN_NEVER_MIGRABLE
            assert entry["approval_status"] in gate.APPROVAL_STATES
