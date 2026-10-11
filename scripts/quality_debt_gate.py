#!/usr/bin/env python3
"""Strict Option-R quality-debt ratchet.

Runs the real Ruff / format / isort / codespell scans. Historical findings on
SHA-pinned files may remain. Any new finding, SHA drift, rename, or diagnostic
identity change fails. A zero exit means BASELINE_ONLY accepted, not a clean tree.

QUALITY-3 extension (minimal, backwards compatible): an optional machine-readable
migration manifest (``quality/approved_migrations.json``) can authorize exact
old->new SHA-256 transitions of baseline-pinned files and proven diagnostic
removals / row shifts. Entries default to ``PENDING_OWNER``; only owner-committed
``APPROVED`` entries with full 64-hex SHA pairs (never prefixes or wildcards) are
honored. Frozen physics sources can never be migrated. Without a manifest the
gate behaves exactly as the exact-snapshot freeze it was.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BASELINE_NAME = "legacy_two_phase_debt_v1"
REQUIRED_SUMMARY = {
    "ruff_findings": 171,
    "format_files": 27,
    "isort_files": 28,
    "codespell_findings": 3,
}
# Whole-file seal of quality/legacy_two_phase_debt_v1.json. Editing the baseline
# (summary counts, records, or pinned SHAs) without a separately reviewed change
# to this constant fails the gate; it can never be bypassed from data files alone.
BASELINE_SHA256 = "52b41ab68b4d6b86aaa254389eb5398e65ad8c5cbdcd12626c0466248dd3d70b"

MIGRATIONS_NAME = "approved_migrations.json"
MIGRATIONS_VERSION = "quality_migrations_v1"
CHANGE_CLASSES = frozenset({"LEGACY_UNCHANGED", "LEGACY_REMOVED_APPROVED", "LEGACY_SHIFTED_APPROVED"})
APPROVAL_STATES = frozenset({"PENDING_OWNER", "APPROVED", "REJECTED"})
# Solver-contract frozen physics sources: SHA drift always fails; QUALITY-3 grants
# no migration path. A contract change requires its own authorized process.
FROZEN_NEVER_MIGRABLE = frozenset(
    {
        "examples/two_phase/phasefield.py",
        "examples/two_phase/generate_dataset.py",
        "examples/two_phase/cases.py",
    }
)
_DIAG_TOOLS = ("ruff", "format", "isort", "codespell")


def repo_root_from(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "quality").is_dir():
            return candidate
    raise SystemExit("GATE_FAIL: cannot locate repository root")


def load_baseline(root: Path, path: Path | None = None) -> dict[str, Any]:
    baseline_path = path or (root / "quality" / "legacy_two_phase_debt_v1.json")
    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"GATE_FAIL: baseline unreadable: {exc}") from exc
    if data.get("version") != BASELINE_NAME:
        raise SystemExit("GATE_FAIL: baseline version mismatch")
    if data.get("policy") != "option_R_non_increasing_exact_multiset":
        raise SystemExit("GATE_FAIL: baseline policy mismatch")
    summary = data.get("summary") or {}
    for key, expected in REQUIRED_SUMMARY.items():
        if int(summary.get(key, -1)) != expected:
            raise SystemExit(f"GATE_FAIL: baseline summary {key} != {expected}")
    if len(data.get("ruff") or []) != REQUIRED_SUMMARY["ruff_findings"]:
        raise SystemExit("GATE_FAIL: baseline ruff record count mismatch")
    if len(data.get("format_files") or []) != REQUIRED_SUMMARY["format_files"]:
        raise SystemExit("GATE_FAIL: baseline format record count mismatch")
    if len(data.get("isort_files") or []) != REQUIRED_SUMMARY["isort_files"]:
        raise SystemExit("GATE_FAIL: baseline isort record count mismatch")
    if len(data.get("codespell") or []) != REQUIRED_SUMMARY["codespell_findings"]:
        raise SystemExit("GATE_FAIL: baseline codespell record count mismatch")
    files = data.get("file_sha256") or {}
    if not isinstance(files, dict) or not files:
        raise SystemExit("GATE_FAIL: baseline file_sha256 missing")
    for rel, digest in files.items():
        if not isinstance(rel, str) or rel.startswith("/") or ".." in rel.split("/"):
            raise SystemExit(f"GATE_FAIL: illegal baseline path {rel!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise SystemExit(f"GATE_FAIL: illegal sha for {rel}")
    seen = set()
    for item in data.get("ruff") or []:
        ident = ruff_key(item)
        if ident in seen:
            raise SystemExit("GATE_FAIL: duplicate ruff baseline record")
        seen.add(ident)
    digest = sha256_file(baseline_path)
    if digest != BASELINE_SHA256:
        raise SystemExit(f"GATE_FAIL: baseline file sha256 mismatch: {digest} (unreviewed baseline edit)")
    return data


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def posix_rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def ruff_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (item["path"], item["code"], int(item["row"]), int(item["col"]), item["message"])


def codespell_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (item["path"], int(item["row"]), item["token"])


def _validate_entry_diagnostics(entry: dict[str, Any], rel: str, baseline: dict[str, Any]) -> None:
    """Validate old/mapped/new diagnostic descriptors of one migration entry."""
    change_class = entry["change_class"]
    descriptors: dict[str, Any] = {}
    for name in ("old_diagnostics", "mapped_diagnostics", "new_diagnostics"):
        block = entry.get(name)
        if not isinstance(block, dict):
            raise SystemExit(f"GATE_FAIL: migration {name} not an object for {rel}")
        for tool in block:
            if tool not in _DIAG_TOOLS:
                raise SystemExit(f"GATE_FAIL: migration {name} unknown tool {tool!r} for {rel}")
        descriptors[name] = block
    old, mapped, new = descriptors["old_diagnostics"], descriptors["mapped_diagnostics"], descriptors["new_diagnostics"]

    def records(block: dict[str, Any], tool: str) -> list[dict[str, Any]]:
        items = block.get(tool) or []
        if not isinstance(items, list):
            raise SystemExit(f"GATE_FAIL: migration diagnostics {tool} not a list for {rel}")
        for item in items:
            if not isinstance(item, dict):
                raise SystemExit(f"GATE_FAIL: migration diagnostics {tool} record not an object for {rel}")
            if item.get("path") != rel:
                raise SystemExit(f"GATE_FAIL: migration diagnostics {tool} record path != entry path for {rel}")
        return items

    def files(block: dict[str, Any], tool: str) -> list[str]:
        items = block.get(tool) or []
        if not isinstance(items, list) or any(x != rel for x in items):
            raise SystemExit(f"GATE_FAIL: migration {tool} file list must be [] or [entry path] for {rel}")
        return list(items)

    base_ruff = {ruff_key(x) for x in baseline.get("ruff") or []}
    base_codespell = {codespell_key(x) for x in baseline.get("codespell") or []}
    base_format = set(baseline.get("format_files") or [])
    base_isort = set(baseline.get("isort_files") or [])

    for tool in ("ruff", "codespell"):
        old_items = records(old, tool)
        mapped_items = records(mapped, tool)
        new_items = records(new, tool)
        keyfn = ruff_key if tool == "ruff" else codespell_key
        old_keys = {keyfn(x) for x in old_items}
        mapped_keys = {keyfn(x) for x in mapped_items}
        new_keys = {keyfn(x) for x in new_items}
        base_keys = base_ruff if tool == "ruff" else base_codespell
        if not old_keys <= base_keys:
            raise SystemExit(f"GATE_FAIL: migration old {tool} diagnostics not present in baseline for {rel}")
        if tool == "ruff":
            for item in old_items:
                if not isinstance(item.get("code"), str) or not isinstance(item.get("message"), str):
                    raise SystemExit(f"GATE_FAIL: migration ruff record missing code/message for {rel}")
        if change_class == "LEGACY_UNCHANGED":
            if mapped_keys != old_keys or new_keys:
                raise SystemExit(f"GATE_FAIL: LEGACY_UNCHANGED requires mapped == old and empty new {tool} for {rel}")
        elif change_class == "LEGACY_REMOVED_APPROVED":
            if mapped_keys or new_keys:
                raise SystemExit(f"GATE_FAIL: LEGACY_REMOVED_APPROVED requires empty mapped/new {tool} for {rel}")
        elif change_class == "LEGACY_SHIFTED_APPROVED":
            if new_keys:
                raise SystemExit(f"GATE_FAIL: LEGACY_SHIFTED_APPROVED requires empty new {tool} for {rel}")
            if len(mapped_items) != len(old_items):
                raise SystemExit(f"GATE_FAIL: shifted {tool} mapping count mismatch for {rel}")
            for o, m in zip(
                sorted(old_items, key=lambda x: (int(x["row"]), int(x.get("col", 0)))),
                sorted(mapped_items, key=lambda x: (int(x["row"]), int(x.get("col", 0)))),
            ):
                # Same historical code text (fingerprint), only the row may move.
                if not isinstance(o.get("line_text"), str) or o["line_text"] != m.get("line_text"):
                    raise SystemExit(f"GATE_FAIL: shifted {tool} line_text fingerprint mismatch for {rel}")
                if tool == "ruff" and (o["code"], o["col"], o["message"]) != (m["code"], m["col"], m["message"]):
                    raise SystemExit(f"GATE_FAIL: shifted ruff identity mismatch for {rel}")
                if mapped_keys & base_keys:
                    raise SystemExit(f"GATE_FAIL: shifted {tool} target already in baseline for {rel}")
    for tool in ("format", "isort"):
        old_files = files(old, tool)
        mapped_files = files(mapped, tool)
        new_files = files(new, tool)
        base_files = base_format if tool == "format" else base_isort
        if old_files and rel not in base_files:
            raise SystemExit(f"GATE_FAIL: migration old {tool} flag not present in baseline for {rel}")
        if not old_files and rel in base_files:
            raise SystemExit(f"GATE_FAIL: migration old {tool} flag missing but baseline lists {rel}")
        if change_class == "LEGACY_UNCHANGED" and (mapped_files != old_files or new_files != old_files):
            raise SystemExit(f"GATE_FAIL: LEGACY_UNCHANGED requires identical {tool} flags for {rel}")
        if change_class == "LEGACY_REMOVED_APPROVED" and (mapped_files or new_files):
            raise SystemExit(f"GATE_FAIL: LEGACY_REMOVED_APPROVED requires empty mapped/new {tool} for {rel}")
        if change_class == "LEGACY_SHIFTED_APPROVED" and (mapped_files != old_files or new_files):
            raise SystemExit(f"GATE_FAIL: LEGACY_SHIFTED_APPROVED requires mapped old and empty new {tool} for {rel}")


def load_migrations(root: Path, baseline: dict[str, Any], path: Path | None = None) -> list[dict[str, Any]]:
    """Load and strictly validate quality/approved_migrations.json (optional)."""
    migrations_path = path or (root / "quality" / MIGRATIONS_NAME)
    if not Path(migrations_path).is_file():
        return []
    try:
        data = json.loads(Path(migrations_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"GATE_FAIL: migrations unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("GATE_FAIL: migrations root is not an object")
    if data.get("version") != MIGRATIONS_VERSION:
        raise SystemExit("GATE_FAIL: migrations version mismatch")
    if data.get("baseline_ref") != BASELINE_NAME:
        raise SystemExit("GATE_FAIL: migrations baseline_ref mismatch")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise SystemExit("GATE_FAIL: migrations entries is not a list")
    pinned: dict[str, str] = baseline.get("file_sha256") or {}
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise SystemExit("GATE_FAIL: migration entry is not an object")
        rel = entry.get("path")
        if not isinstance(rel, str) or rel.startswith("/") or ".." in rel.split("/"):
            raise SystemExit(f"GATE_FAIL: illegal migration path {rel!r}")
        if rel in seen:
            raise SystemExit(f"GATE_FAIL: duplicate migration entry {rel}")
        seen.add(rel)
        if rel in FROZEN_NEVER_MIGRABLE:
            raise SystemExit(f"GATE_FAIL: FROZEN_SOURCE_MIGRATION_FORBIDDEN:{rel}")
        if rel not in pinned:
            raise SystemExit(f"GATE_FAIL: migration path not baseline-pinned:{rel}")
        for field in ("old_sha256", "new_sha256", "evidence_hash"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get(field, ""))):
                raise SystemExit(f"GATE_FAIL: migration {field} must be a full lowercase 64-hex sha256 for {rel}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(entry.get("pr22_commit", ""))):
            raise SystemExit(f"GATE_FAIL: migration pr22_commit must be a full 40-hex commit sha for {rel}")
        if entry.get("change_class") not in CHANGE_CLASSES:
            raise SystemExit(f"GATE_FAIL: migration change_class invalid for {rel}")
        if entry.get("approval_status") not in APPROVAL_STATES:
            raise SystemExit(f"GATE_FAIL: migration approval_status invalid for {rel}")
        if entry.get("old_sha256") != pinned[rel]:
            raise SystemExit(f"GATE_FAIL: migration old_sha256 != baseline pin for {rel}")
        if entry.get("old_sha256") == entry.get("new_sha256"):
            raise SystemExit(f"GATE_FAIL: migration old_sha256 == new_sha256 for {rel}")
        if entry["approval_status"] == "APPROVED":
            approval = entry.get("approval") or {}
            if not isinstance(approval, dict) or not str(approval.get("approved_by", "")).strip():
                raise SystemExit(f"GATE_FAIL: APPROVED migration missing approved_by for {rel}")
            if not re.fullmatch(r"[0-9a-f]{40}", str(approval.get("approval_commit", ""))):
                raise SystemExit(f"GATE_FAIL: APPROVED migration missing full 40-hex approval_commit for {rel}")
        _validate_entry_diagnostics(entry, rel, baseline)
    return entries


def approved_entries(migrations: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [e for e in (migrations or []) if e.get("approval_status") == "APPROVED"]


def adjust_expected_records(
    expected: Counter,
    migrations: list[dict[str, Any]] | None,
    tool: str,
    keyfn,
) -> tuple[Counter, int, int]:
    """Apply APPROVED shift/removal mappings to the expected diagnostic multiset."""
    moved = removed = 0
    for entry in approved_entries(migrations):
        change_class = entry["change_class"]
        olds = Counter(keyfn(x) for x in (entry.get("old_diagnostics") or {}).get(tool) or [])
        if change_class == "LEGACY_SHIFTED_APPROVED":
            mapped = Counter(keyfn(x) for x in (entry.get("mapped_diagnostics") or {}).get(tool) or [])
            expected = expected - olds + mapped
            moved += sum(olds.values())
        elif change_class == "LEGACY_REMOVED_APPROVED":
            expected = expected - olds
            removed += sum(olds.values())
    return expected, moved, removed


def adjust_expected_files(expected: Counter, migrations: list[dict[str, Any]] | None, tool: str) -> tuple[Counter, int]:
    removed = 0
    for entry in approved_entries(migrations):
        if entry["change_class"] != "LEGACY_REMOVED_APPROVED":
            continue
        olds = (entry.get("old_diagnostics") or {}).get(tool) or []
        for rel in olds:
            if expected[rel] > 0:
                expected[rel] -= 1
                removed += 1
    return +expected, removed


def run_cmd(root: Path, argv: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    return subprocess.run(
        argv,
        cwd=root,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def uv_run(root: Path, rest: list[str]) -> subprocess.CompletedProcess[str]:
    return run_cmd(root, ["uv", "run", "--no-sync", *rest])


def check_pinned_shas(
    root: Path,
    baseline: dict[str, Any],
    relevant: set[str],
    migrations: list[dict[str, Any]] | None = None,
    transitions_out: list[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    pinned: dict[str, str] = baseline["file_sha256"]
    entries = {e["path"]: e for e in (migrations or [])}
    for rel in sorted(relevant - set(pinned)):
        errors.append(f"NEW_PATH_NOT_IN_BASELINE:{rel}")
    for rel in sorted(pinned):
        path = root / rel
        if not path.is_file():
            errors.append(f"MISSING_PINNED_FILE:{rel}")
            continue
        digest = sha256_file(path)
        entry = entries.get(rel)
        if entry is not None and entry.get("approval_status") == "APPROVED":
            # An approval is an exact old->new transition, not a standing
            # exemption. The working tree must contain the named new digest.
            if entry.get("old_sha256") != pinned[rel] or entry.get("new_sha256") != digest:
                errors.append(f"PROTECTED_SHA_DRIFT_UNAPPROVED:{rel}")
                continue
            if transitions_out is not None:
                transitions_out.append(rel)
            continue
        if digest == pinned[rel]:
            continue
        if entry is not None:
            errors.append(f"PROTECTED_SHA_DRIFT_UNAPPROVED:{rel}")
        else:
            errors.append(f"SHA_DRIFT:{rel}")
    return errors


def to_repo_rel(root: Path, filename: str) -> str:
    path = Path(filename)
    try:
        if path.is_absolute():
            return path.resolve().relative_to(root.resolve()).as_posix()
        return path.as_posix()
    except ValueError:
        return _abs_to_rel(str(path))


def parse_ruff(payload: str, root: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"GATE_FAIL: ruff JSON unparsable: {exc}") from exc
    if not isinstance(data, list):
        raise SystemExit("GATE_FAIL: ruff JSON is not a list")
    out = []
    for x in data:
        loc = x.get("location") or {}
        out.append(
            {
                "path": to_repo_rel(root, str(x.get("filename") or "")),
                "code": x["code"],
                "row": loc.get("row"),
                "col": loc.get("column"),
                "message": x.get("message") or "",
            }
        )
    return out


def _abs_to_rel(filename: str) -> str:
    name = filename.replace("\\", "/")
    for needle in ("/examples/", "/hydrogym/", "/test/", "/docs/", "/scripts/", "/quality/"):
        idx = name.find(needle)
        if idx != -1:
            return name[idx + 1 :]
    return Path(filename).name


def parse_format_files(diff_text: str, root: Path) -> list[str]:
    files: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("--- "):
            raw = line[4:].split("\t")[0].split(":")[0].strip()
            raw = raw.replace("\\", "/")
            if raw.startswith(str(root)):
                raw = str(Path(raw).resolve().relative_to(root)).replace("\\", "/")
            else:
                raw = raw.replace("\\", "/")
                if raw.startswith("/"):
                    raw = _abs_to_rel(raw)
            if raw and raw != "/dev/null":
                files.append(raw)
    return sorted(set(files))


def parse_isort_files(stderr: str, stdout: str, root: Path) -> list[str]:
    text = stderr + "\n" + stdout
    files: list[str] = []
    for line in text.splitlines():
        match = re.match(r"ERROR: (.+) Imports are incorrectly sorted", line)
        if not match:
            continue
        raw = match.group(1).replace("\\", "/")
        if raw.startswith(str(root)):
            raw = str(Path(raw).resolve().relative_to(root)).replace("\\", "/")
        files.append(raw)
    return sorted(set(files))


def parse_codespell(text: str) -> list[dict[str, Any]]:
    out = []
    for line in text.splitlines():
        match = re.match(r"(.+):(\d+): (\S+) ==> (.+)$", line)
        if not match:
            continue
        out.append(
            {
                "path": match.group(1).replace("\\", "/"),
                "row": int(match.group(2)),
                "token": match.group(3),
                "suggestion": match.group(4),
                "raw": line,
            }
        )
    return out


def compare_multiset(actual: Counter, expected: Counter, label: str) -> list[str]:
    errors: list[str] = []
    extra = actual - expected
    missing = expected - actual
    for key, count in extra.items():
        errors.append(f"NEW_FINDING:{label}:{key}x{count}")
    for key, count in missing.items():
        errors.append(f"MISSING_HISTORICAL:{label}:{key}x{count}")
    return errors


def emit_result(
    tool: str,
    raw_exit: int,
    historical: int,
    new_findings: int,
    gate_ok: bool,
    extra: list[str],
    ratchet_note: str = "",
) -> int:
    status = "PASS_BASELINE_ONLY" if gate_ok else "FAIL"
    raw_label = "RAW_TOOL_FAIL" if raw_exit else "RAW_TOOL_OK"
    print(f"{raw_label} tool={tool} RAW_TOOL_EXIT={raw_exit}")
    print(f"HISTORICAL_EXCEPTIONS={historical}; NEW_FINDINGS={new_findings}; GATE={status}")
    if ratchet_note:
        print(f"RATCHET_DECISION={status} {ratchet_note}")
    if extra:
        for line in extra[:50]:
            print(line)
        if len(extra) > 50:
            print(f"... {len(extra) - 50} more")
    if gate_ok:
        print("BASELINE_ONLY_ACCEPTED (historical debt remains; not a clean tree)")
        return 0
    print("GATE_FAIL")
    return 1


def gate_ruff(root: Path, baseline: dict[str, Any], migrations: list[dict[str, Any]] | None = None) -> int:
    proc = uv_run(root, ["ruff", "check", ".", "--output-format", "json"])
    if proc.returncode not in (0, 1):
        print(proc.stderr)
        return emit_result("ruff", proc.returncode, 0, 0, False, ["TOOL_CRASH"])
    findings = parse_ruff(proc.stdout or "[]", root)
    actual = Counter(ruff_key(x) for x in findings)
    expected = Counter(ruff_key(x) for x in baseline["ruff"])
    expected, moved, removed = adjust_expected_records(expected, migrations, "ruff", ruff_key)
    extra = compare_multiset(actual, expected, "ruff")
    transitions: list[str] = []
    extra.extend(check_pinned_shas(root, baseline, {x["path"] for x in findings}, migrations, transitions))
    new_n = sum(1 for e in extra if e.startswith("NEW_"))
    ok = not extra
    return emit_result(
        "ruff",
        proc.returncode,
        len(expected),
        new_n,
        ok,
        extra,
        _ratchet_note(actual, expected, extra, moved, removed, transitions),
    )


def gate_format(root: Path, baseline: dict[str, Any], migrations: list[dict[str, Any]] | None = None) -> int:
    proc = uv_run(root, ["ruff", "format", "--check", "--diff", "."])
    if proc.returncode not in (0, 1):
        print(proc.stderr)
        return emit_result("format", proc.returncode, 0, 0, False, ["TOOL_CRASH"])
    files = parse_format_files((proc.stdout or "") + (proc.stderr or ""), root)
    actual = Counter(files)
    expected = Counter(baseline["format_files"])
    expected, removed = adjust_expected_files(expected, migrations, "format")
    extra = compare_multiset(actual, expected, "format")
    transitions: list[str] = []
    extra.extend(check_pinned_shas(root, baseline, set(files), migrations, transitions))
    new_n = sum(1 for e in extra if e.startswith("NEW_"))
    ok = not extra
    return emit_result(
        "format",
        proc.returncode,
        len(expected),
        new_n,
        ok,
        extra,
        _ratchet_note(actual, expected, extra, 0, removed, transitions),
    )


def gate_isort(root: Path, baseline: dict[str, Any], migrations: list[dict[str, Any]] | None = None) -> int:
    proc = uv_run(root, ["isort", ".", "--check-only", "--diff"])
    if proc.returncode not in (0, 1):
        print(proc.stderr)
        return emit_result("isort", proc.returncode, 0, 0, False, ["TOOL_CRASH"])
    files = parse_isort_files(proc.stderr or "", proc.stdout or "", root)
    actual = Counter(files)
    expected = Counter(baseline["isort_files"])
    expected, removed = adjust_expected_files(expected, migrations, "isort")
    extra = compare_multiset(actual, expected, "isort")
    transitions: list[str] = []
    extra.extend(check_pinned_shas(root, baseline, set(files), migrations, transitions))
    new_n = sum(1 for e in extra if e.startswith("NEW_"))
    ok = not extra
    return emit_result(
        "isort",
        proc.returncode,
        len(expected),
        new_n,
        ok,
        extra,
        _ratchet_note(actual, expected, extra, 0, removed, transitions),
    )


def gate_codespell(root: Path, baseline: dict[str, Any], migrations: list[dict[str, Any]] | None = None) -> int:
    argv = [
        "codespell",
        "--toml",
        "pyproject.toml",
        "README.md",
        "CONTRIBUTING.md",
        "docs",
        "examples",
        "test",
        "hydrogym",
    ]
    proc = uv_run(root, argv)
    if proc.returncode not in (0, 65):
        print(proc.stderr)
        print(proc.stdout)
        return emit_result("codespell", proc.returncode, 0, 0, False, ["TOOL_CRASH"])
    findings = parse_codespell((proc.stdout or "") + (proc.stderr or ""))
    actual = Counter(codespell_key(x) for x in findings)
    expected = Counter(codespell_key(x) for x in baseline["codespell"])
    expected, moved, removed = adjust_expected_records(expected, migrations, "codespell", codespell_key)
    extra = compare_multiset(actual, expected, "codespell")
    transitions: list[str] = []
    extra.extend(check_pinned_shas(root, baseline, {x["path"] for x in findings}, migrations, transitions))
    new_n = sum(1 for e in extra if e.startswith("NEW_"))
    ok = not extra
    return emit_result(
        "codespell",
        proc.returncode,
        len(expected),
        new_n,
        ok,
        extra,
        _ratchet_note(actual, expected, extra, moved, removed, transitions),
    )


GATES = {
    "ruff": gate_ruff,
    "format": gate_format,
    "isort": gate_isort,
    "codespell": gate_codespell,
}


def _ratchet_note(
    actual: Counter,
    expected: Counter,
    extra: list[str],
    moved: int,
    removed: int,
    transitions: list[str],
) -> str:
    kept = sum((actual & expected).values())
    added = sum((actual - expected).values())
    missing = sum((expected - actual).values())
    unapproved = sum(1 for e in extra if e.startswith(("SHA_DRIFT", "PROTECTED_SHA_DRIFT_UNAPPROVED")))
    return (
        f"kept={kept} added={added} missing_historical={missing} moved_approved={moved} "
        f"removed_approved={removed} sha_transitions_approved={len(transitions)} sha_drift_unapproved={unapproved}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict legacy quality-debt gate")
    parser.add_argument("--tool", required=True, choices=[*GATES, "all"])
    parser.add_argument("--root", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--migrations", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else repo_root_from()
    baseline = load_baseline(root, Path(args.baseline) if args.baseline else None)
    migrations = load_migrations(root, baseline, Path(args.migrations) if args.migrations else None)
    n_approved = len(approved_entries(migrations))
    print(
        f"MIGRATIONS_LOADED={len(migrations)} APPROVED={n_approved} PENDING_OR_REJECTED={len(migrations) - n_approved}"
    )
    tools = list(GATES) if args.tool == "all" else [args.tool]
    overall = 0
    for tool in tools:
        code = GATES[tool](root, baseline, migrations)
        overall = overall or code
    return overall


if __name__ == "__main__":
    sys.exit(main())
