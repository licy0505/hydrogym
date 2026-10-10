#!/usr/bin/env python3
"""QUALITY-1 protocol checks. Not a physics test. Fail closed."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
INV = json.loads((pathlib.Path(__file__).with_name("frozen_source_inventory.json")).read_text())
BASE = json.loads((pathlib.Path(__file__).with_name("quality_baseline.json")).read_text())
CI = json.loads((pathlib.Path(__file__).with_name("ci_validation.json")).read_text())


def sha256(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main() -> int:
    errors: list[str] = []
    for rel, expected in INV["tier_P"]["sha256"].items():
        got = sha256(rel)
        if got != expected:
            errors.append(f"protected hash drift {rel}")
    if CI["quality_hosted_pr23"].get("format") == "SKIPPED" and CI["quality_hosted_pr23"].get("skipped_not_success") is not True:
        errors.append("SKIPPED must not be treated as success")
    if BASE["commands"]["ruff_check"]["verdict"] != "FAIL":
        errors.append("baseline ruff must remain FAIL until a real fix")
    if BASE["tier_u_safe_fixes_available"] != 0:
        errors.append("unexpected Tier-U claim")
    if any(not p.startswith("examples/two_phase/") for p in BASE["ruff_by_file"]):
        errors.append("ruff leaked outside two_phase")
    green_forbidden = (
        CI["quality_local"]["ruff_check"] != "PASS"
        or CI["quality_hosted_pr23"]["lint"] != "PASS"
        or CI["quality_hosted_pr23"]["format"] == "SKIPPED"
    )
    if not green_forbidden:
        errors.append("protocol should forbid QUALITY1_GREEN under current CI")
    if errors:
        print("FAIL", errors)
        return 1
    print("PASS quality1 invariants (protected hashes unchanged; no green verdict)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
