#!/usr/bin/env python3
"""Strict Option-R quality-debt ratchet.

Runs the real Ruff / format / isort / codespell scans. Historical findings on
SHA-pinned files may remain. Any new finding, SHA drift, rename, or diagnostic
identity change fails. A zero exit means BASELINE_ONLY accepted, not a clean tree.
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
    return data


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def posix_rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def ruff_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (item["path"], item["code"], int(item["row"]), int(item["col"]), item["message"])


def codespell_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (item["path"], int(item["row"]), item["token"])


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


def check_pinned_shas(root: Path, baseline: dict[str, Any], relevant: set[str]) -> list[str]:
    errors: list[str] = []
    pinned: dict[str, str] = baseline["file_sha256"]
    for rel in sorted(relevant - set(pinned)):
        errors.append(f"NEW_PATH_NOT_IN_BASELINE:{rel}")
    for rel in sorted(pinned):
        path = root / rel
        if not path.is_file():
            errors.append(f"MISSING_PINNED_FILE:{rel}")
            continue
        digest = sha256_file(path)
        if digest != pinned[rel]:
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
) -> int:
    status = "PASS_BASELINE_ONLY" if gate_ok else "FAIL"
    raw_label = "RAW_TOOL_FAIL" if raw_exit else "RAW_TOOL_OK"
    print(f"{raw_label} tool={tool} RAW_TOOL_EXIT={raw_exit}")
    print(f"HISTORICAL_EXCEPTIONS={historical}; NEW_FINDINGS={new_findings}; GATE={status}")
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


def gate_ruff(root: Path, baseline: dict[str, Any]) -> int:
    proc = uv_run(root, ["ruff", "check", ".", "--output-format", "json"])
    if proc.returncode not in (0, 1):
        print(proc.stderr)
        return emit_result("ruff", proc.returncode, 0, 0, False, ["TOOL_CRASH"])
    findings = parse_ruff(proc.stdout or "[]", root)
    actual = Counter(ruff_key(x) for x in findings)
    expected = Counter(ruff_key(x) for x in baseline["ruff"])
    extra = compare_multiset(actual, expected, "ruff")
    extra.extend(check_pinned_shas(root, baseline, {x["path"] for x in findings}))
    new_n = sum(1 for e in extra if e.startswith("NEW_"))
    ok = not extra
    return emit_result("ruff", proc.returncode, len(expected), new_n, ok, extra)


def gate_format(root: Path, baseline: dict[str, Any]) -> int:
    proc = uv_run(root, ["ruff", "format", "--check", "--diff", "."])
    if proc.returncode not in (0, 1):
        print(proc.stderr)
        return emit_result("format", proc.returncode, 0, 0, False, ["TOOL_CRASH"])
    files = parse_format_files((proc.stdout or "") + (proc.stderr or ""), root)
    actual = Counter(files)
    expected = Counter(baseline["format_files"])
    extra = compare_multiset(actual, expected, "format")
    extra.extend(check_pinned_shas(root, baseline, set(files)))
    new_n = sum(1 for e in extra if e.startswith("NEW_"))
    ok = not extra
    return emit_result("format", proc.returncode, len(expected), new_n, ok, extra)


def gate_isort(root: Path, baseline: dict[str, Any]) -> int:
    proc = uv_run(root, ["isort", ".", "--check-only", "--diff"])
    if proc.returncode not in (0, 1):
        print(proc.stderr)
        return emit_result("isort", proc.returncode, 0, 0, False, ["TOOL_CRASH"])
    files = parse_isort_files(proc.stderr or "", proc.stdout or "", root)
    actual = Counter(files)
    expected = Counter(baseline["isort_files"])
    extra = compare_multiset(actual, expected, "isort")
    extra.extend(check_pinned_shas(root, baseline, set(files)))
    new_n = sum(1 for e in extra if e.startswith("NEW_"))
    ok = not extra
    return emit_result("isort", proc.returncode, len(expected), new_n, ok, extra)


def gate_codespell(root: Path, baseline: dict[str, Any]) -> int:
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
    extra = compare_multiset(actual, expected, "codespell")
    extra.extend(check_pinned_shas(root, baseline, {x["path"] for x in findings}))
    new_n = sum(1 for e in extra if e.startswith("NEW_"))
    ok = not extra
    return emit_result("codespell", proc.returncode, len(expected), new_n, ok, extra)


GATES = {
    "ruff": gate_ruff,
    "format": gate_format,
    "isort": gate_isort,
    "codespell": gate_codespell,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict legacy quality-debt gate")
    parser.add_argument("--tool", required=True, choices=[*GATES, "all"])
    parser.add_argument("--root", default=None)
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else repo_root_from()
    baseline = load_baseline(root, Path(args.baseline) if args.baseline else None)
    tools = list(GATES) if args.tool == "all" else [args.tool]
    overall = 0
    for tool in tools:
        code = GATES[tool](root, baseline)
        overall = overall or code
    return overall


if __name__ == "__main__":
    sys.exit(main())
