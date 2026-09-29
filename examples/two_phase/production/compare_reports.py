"""Strict before/after comparison of two L1A validation reports (L1A-2a evidence tool).

Both inputs are ``report.json`` files written by ``production.run_validation``.  The tool is
read-only and does not import JAX.  It tabulates the static-droplet Laplace response, parasitic
currents, mass drift, sessile contact angles and flat-wall impact observables, and evaluates the
L1A-2a guards.  A failed guard means STOP and investigate; guards must never be relaxed to make a
run pass.  Contact-angle and pre-contact numbers are *recorded, not judged*: wetting is out of scope.

Run from ``examples/two_phase``::

    python -m production.compare_reports --before before/report.json --after after/report.json \\
        [--markdown out.md] [--json out.json] [--wet-band 0.15]

Exit status: 0 = every guard passed, 1 = a guard failed (STOP), 2 = unusable input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from production.report import validate_report_schema

# --- guard thresholds (stated up front; they are screening rules, not tuned to a result) ---------
LAPLACE_MIN_R_SQUARED = 0.99  # delta_p vs 1/R fit must stay linear in curvature
LAPLACE_TARGET = 0.05  # provisional P1 target, mirrors PROVISIONAL_READINESS_TARGETS
RESOLVED_RADIUS_MIN = 0.8  # radii >= this are "resolved" at N=128, eps=2 dx
SPURIOUS_GROWTH_LIMIT = 5.0  # after peak <= max(5 x before, SPURIOUS_ABS_FLOOR)
SPURIOUS_ABS_FLOOR = 1.0e-3
STATIC_MASS_DRIFT_LIMIT = 1.0e-4
IMPACT_MASS_GROWTH_LIMIT = 5.0  # after drift <= max(5 x before, IMPACT_MASS_ABS_FLOOR)
IMPACT_MASS_ABS_FLOOR = 1.0e-3
IMPACT_SPEED_GROWTH_LIMIT = 2.0  # no explosive acceleration beyond 2 x the before peak speed
UNTOUCHED_BLOCKERS = ("P-VARDENS-PROJ", "P-CAP-RHO", "N-DT", "P-VARVISC", "BC-Y-PERIODIC")
PROMOTED_STATUSES = ("VALIDATED", "PRODUCTION_READY")


def _static_cases(report: dict[str, Any]) -> dict[float, dict[str, Any]]:
    cases = report["benchmarks"]["static_droplet"].get("cases", [])
    bad = [c for c in cases if c.get("finite") is not True]
    if bad:
        raise ValueError(f"static_droplet contains non-finite / failed cases: {[c.get('R') for c in bad]}")
    return {round(float(c["R"]), 9): c for c in cases}


def _impact_cases(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(c["case_name"]): c for c in report["benchmarks"]["impact"].get("cases", [])}


def _contact_cases(report: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {round(float(c["target_deg"]), 9): c for c in report["benchmarks"]["contact_angle"].get("cases", [])}


def _guard(name: str, passed: bool, detail: str, kind: str = "guard") -> dict[str, Any]:
    return {"name": name, "kind": kind, "passed": bool(passed), "detail": detail}


def compare_reports(before: dict[str, Any], after: dict[str, Any], wet_band: float = 0.15) -> dict[str, Any]:
    """Return tables and guard verdicts for a strict before -> after comparison."""
    guards: list[dict[str, Any]] = []
    b_repo, a_repo = before["repository"], after["repository"]
    lineage = {
        "git_sha": (b_repo["git_sha"], a_repo["git_sha"]),
        "solver_contract_version": (b_repo["solver_contract_version"], a_repo["solver_contract_version"]),
        "phasefield_sha256": (b_repo["phasefield_sha256"], a_repo["phasefield_sha256"]),
        "validation_code_sha256": (b_repo["validation_code_sha256"], a_repo["validation_code_sha256"]),
        "config_sha256": (before["config"]["sha256"], after["config"]["sha256"]),
        "physics_status": (before["physics_status"], after["physics_status"]),
    }
    guards.append(
        _guard(
            "baseline_config_unchanged",
            lineage["config_sha256"][0] == lineage["config_sha256"][1],
            "both runs must use the identical baseline config (strict before -> one change -> after)",
        )
    )
    guards.append(
        _guard(
            "solver_contract_bumped",
            a_repo["solver_contract_version"] > b_repo["solver_contract_version"],
            f"contract {b_repo['solver_contract_version']} -> {a_repo['solver_contract_version']}",
        )
    )
    guards.append(
        _guard(
            "physics_status_not_promoted",
            after["physics_status"] not in PROMOTED_STATUSES,
            f"after physics_status = {after['physics_status']} (never VALIDATED / PRODUCTION_READY here)",
        )
    )

    # ---- static droplet: Laplace response, parasitic currents, mass ---------------------------
    b_static, a_static = _static_cases(before), _static_cases(after)
    if set(b_static) != set(a_static):
        raise ValueError(f"static radii differ: before={sorted(b_static)} after={sorted(a_static)}")
    static_rows = []
    for radius in sorted(a_static):
        b, a = b_static[radius], a_static[radius]
        static_rows.append(
            {
                "R": radius,
                "before_ratio": float(b["laplace_ratio"]),
                "after_ratio": float(a["laplace_ratio"]),
                "abs_error_after": abs(float(a["laplace_ratio"]) - 1.0),
                "before_delta_p": float(b["delta_p"]),
                "after_delta_p": float(a["delta_p"]),
                "before_peak_speed": float(b["max_speed_peak"]),
                "after_peak_speed": float(a["max_speed_peak"]),
                "before_final_speed": float(b["max_speed_final"]),
                "after_final_speed": float(a["max_speed_final"]),
                "before_kinetic_energy_final": float(b["kinetic_energy_final"]),
                "after_kinetic_energy_final": float(a["kinetic_energy_final"]),
                "before_mass_drift": float(b["mass_relative_drift"]),
                "after_mass_drift": float(a["mass_relative_drift"]),
            }
        )
    b_fit = before["benchmarks"]["static_droplet"]["summary"]
    a_fit = after["benchmarks"]["static_droplet"]["summary"]
    errors = [row["abs_error_after"] for row in static_rows]
    laplace = {
        "rows": static_rows,
        "before_slope": float(b_fit["slope_delta_p_vs_inv_R"]),
        "after_slope": float(a_fit["slope_delta_p_vs_inv_R"]),
        "before_r_squared": b_fit["r_squared"],
        "after_r_squared": a_fit["r_squared"],
        "after_mean_abs_error": sum(errors) / len(errors),
        "after_max_abs_error": max(errors),
    }
    positive = all(row["after_ratio"] > 0.0 for row in static_rows)
    guards.append(
        _guard(
            "P0_all_laplace_ratios_positive",
            positive,
            "after ratios: " + ", ".join(f"{row['after_ratio']:+.4f}" for row in static_rows),
        )
    )
    r2 = a_fit["r_squared"]
    guards.append(
        _guard(
            "P0_delta_p_vs_inv_R_scaling",
            r2 is not None and float(r2) >= LAPLACE_MIN_R_SQUARED and laplace["after_slope"] > 0.0,
            f"R^2 = {r2}, slope = {laplace['after_slope']:+.6g} (need R^2 >= {LAPLACE_MIN_R_SQUARED}, slope > 0)",
        )
    )
    resolved = [row for row in static_rows if row["R"] >= RESOLVED_RADIUS_MIN]
    guards.append(
        _guard(
            "P1_resolved_radii_within_provisional_target",
            bool(resolved) and all(row["abs_error_after"] <= LAPLACE_TARGET for row in resolved),
            f"|ratio-1| <= {LAPLACE_TARGET} for R >= {RESOLVED_RADIUS_MIN}: "
            + ", ".join(f"R={_radius(row['R'])}: {row['abs_error_after']:.4f}" for row in resolved),
            kind="target",
        )
    )
    guards.append(
        _guard(
            "P1_all_radii_within_provisional_target",
            all(error <= LAPLACE_TARGET for error in errors),
            f"max |ratio-1| = {laplace['after_max_abs_error']:.4f} (target {LAPLACE_TARGET})",
            kind="target",
        )
    )
    spurious_ok = all(
        row["after_peak_speed"] <= max(SPURIOUS_GROWTH_LIMIT * row["before_peak_speed"], SPURIOUS_ABS_FLOOR)
        for row in static_rows
    )
    guards.append(
        _guard(
            "spurious_current_peak_guard",
            spurious_ok,
            f"after peak <= max({SPURIOUS_GROWTH_LIMIT:g} x before, {SPURIOUS_ABS_FLOOR:g}) for every R",
        )
    )
    guards.append(
        _guard(
            "static_mass_drift_guard",
            all(row["after_mass_drift"] < STATIC_MASS_DRIFT_LIMIT for row in static_rows),
            f"after static mass drift < {STATIC_MASS_DRIFT_LIMIT:g}: "
            + ", ".join(f"{row['after_mass_drift']:.2e}" for row in static_rows),
        )
    )

    # ---- contact angle (recorded, not judged) -------------------------------------------------
    b_contact, a_contact = _contact_cases(before), _contact_cases(after)
    contact_rows = []
    for target in sorted(set(b_contact) & set(a_contact)):
        b, a = b_contact[target], a_contact[target]
        contact_rows.append(
            {
                "target_deg": target,
                "before_measured_deg": b.get("measured_deg"),
                "after_measured_deg": a.get("measured_deg"),
                "before_error_deg": b.get("signed_error_deg"),
                "after_error_deg": a.get("signed_error_deg"),
            }
        )
    b_angle = before["benchmarks"]["contact_angle"].get("summary") or {}
    a_angle = after["benchmarks"]["contact_angle"].get("summary") or {}
    contact = {
        "rows": contact_rows,
        "before_mae_deg": b_angle.get("mae_deg"),
        "after_mae_deg": a_angle.get("mae_deg"),
        "before_max_abs_error_deg": b_angle.get("max_absolute_error_deg"),
        "after_max_abs_error_deg": a_angle.get("max_absolute_error_deg"),
    }

    # ---- impact (flat wall) -------------------------------------------------------------------
    b_impact, a_impact = _impact_cases(before), _impact_cases(after)
    if set(b_impact) != set(a_impact):
        raise ValueError(f"impact cases differ: before={sorted(b_impact)} after={sorted(a_impact)}")
    impact_rows = []
    for name in a_impact:  # keep the config order of the report
        b, a = b_impact[name], a_impact[name]
        row: dict[str, Any] = {"case_name": name, "before_finite": b.get("finite"), "after_finite": a.get("finite")}
        if b.get("finite") is True and a.get("finite") is True:
            for label, case in (("before", b), ("after", a)):
                series = case["time_series"]
                row[f"{label}_beta_max"] = float(case["beta_max"])
                row[f"{label}_final_y_cm"] = float(case["final_y_cm"])
                row[f"{label}_min_g05"] = float(min(series["g_0.5"]))
                row[f"{label}_min_g01"] = float(min(series["g_0.1"]))
                row[f"{label}_mass_drift"] = float(case["mass_drift"])
                row[f"{label}_peak_speed"] = float(max(series["max_speed"]))
                row[f"{label}_final_speed"] = float(series["max_speed"][-1])
                row[f"{label}_first_contact"] = case.get("first_contact_time")
            row["after_min_g05_over_wet_band"] = row["after_min_g05"] / float(wet_band)
        impact_rows.append(row)
    all_finite = bool(impact_rows) and all(
        r["before_finite"] is True and r["after_finite"] is True for r in impact_rows
    )
    guards.append(_guard("impact_trajectories_finite", all_finite, f"{len(impact_rows)} impact case(s)"))
    if all_finite:
        guards.append(
            _guard(
                "impact_mass_drift_guard",
                all(
                    r["after_mass_drift"]
                    <= max(IMPACT_MASS_GROWTH_LIMIT * r["before_mass_drift"], IMPACT_MASS_ABS_FLOOR)
                    for r in impact_rows
                ),
                f"after drift <= max({IMPACT_MASS_GROWTH_LIMIT:g} x before, {IMPACT_MASS_ABS_FLOOR:g})",
            )
        )
        guards.append(
            _guard(
                "impact_no_explosive_acceleration_guard",
                all(r["after_peak_speed"] <= IMPACT_SPEED_GROWTH_LIMIT * r["before_peak_speed"] for r in impact_rows),
                f"after peak speed <= {IMPACT_SPEED_GROWTH_LIMIT:g} x before peak speed",
            )
        )

    # ---- blockers -----------------------------------------------------------------------------
    b_block = {item["id"]: item["status"] for item in before["known_solver_blockers"]}
    a_block = {item["id"]: item["status"] for item in after["known_solver_blockers"]}
    blockers = {
        name: (b_block.get(name), a_block.get(name))
        for name in list(a_block) + [n for n in b_block if n not in a_block]
    }
    guards.append(
        _guard(
            "untouched_blockers_remain_open",
            all(a_block.get(name) == "open" for name in UNTOUCHED_BLOCKERS),
            ", ".join(f"{name}={a_block.get(name)}" for name in UNTOUCHED_BLOCKERS),
        )
    )
    closed = a_block.get("P-LAPLACE-SIGN") == "resolved_in_contract_v5"
    guards.append(
        _guard(
            "laplace_blocker_closed_only_with_evidence",
            (not closed) or (positive and r2 is not None and float(r2) >= LAPLACE_MIN_R_SQUARED),
            f"P-LAPLACE-SIGN = {a_block.get('P-LAPLACE-SIGN')}",
        )
    )

    return {
        "lineage": lineage,
        "laplace": laplace,
        "contact_angle": contact,
        "impact": impact_rows,
        "blockers": blockers,
        "guards": guards,
        "guards_passed": all(g["passed"] for g in guards if g["kind"] == "guard"),
    }


# --------------------------------------------------------------------------------------------
#  Markdown rendering
# --------------------------------------------------------------------------------------------


def _radius(value: float) -> str:
    text = f"{value:g}"
    return text if "." in text or "e" in text else text + ".0"


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    return format(value, spec) if isinstance(value, (int, float)) else str(value)


def format_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = ["## L1A-2a before / after (unchanged baseline config)", ""]
    lines += ["| lineage | before | after |", "|---|---|---|"]
    for key, (b, a) in result["lineage"].items():
        shown = (lambda v: f"`{v[:12]}`" if isinstance(v, str) and len(v) >= 40 else f"`{v}`") if "sha" in key else str
        lines.append(f"| {key} | {shown(b)} | {shown(a)} |")
    lam = result["laplace"]
    lines += ["", "### Static droplet Laplace (`laplace_ratio = (p_inside - p_outside) R We`, target +1)", ""]
    lines += ["| R | Before ratio | After ratio | \\|After−1\\| |", "|---|---:|---:|---:|"]
    for row in lam["rows"]:
        lines.append(
            f"| {_radius(row['R'])} | {row['before_ratio']:+.4f} | {row['after_ratio']:+.4f} "
            f"| {row['abs_error_after']:.4f} |"
        )
    lines += [
        "",
        f"- slope of delta_p vs 1/R: before {lam['before_slope']:+.6f}, after {lam['after_slope']:+.6f}",
        f"- R²: before {_fmt(lam['before_r_squared'], '.5f')}, after {_fmt(lam['after_r_squared'], '.5f')}",
        f"- after mean |ratio−1| = {lam['after_mean_abs_error']:.4f}, max |ratio−1| = {lam['after_max_abs_error']:.4f}",
    ]
    lines += ["", "### Parasitic current and mass (static droplet)", ""]
    lines += [
        "| R | peak before | peak after | final before | final after | KE final before | KE final after "
        "| mass drift before | mass drift after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in lam["rows"]:
        lines.append(
            f"| {_radius(row['R'])} | {row['before_peak_speed']:.3e} | {row['after_peak_speed']:.3e} "
            f"| {row['before_final_speed']:.3e} | {row['after_final_speed']:.3e} "
            f"| {row['before_kinetic_energy_final']:.3e} | {row['after_kinetic_energy_final']:.3e} "
            f"| {row['before_mass_drift']:.2e} | {row['after_mass_drift']:.2e} |"
        )
    contact = result["contact_angle"]
    lines += ["", "### Contact angle (recorded only; wetting is NOT fixed or tuned in this PR)", ""]
    lines += ["| target (deg) | before | after | before err | after err |", "|---:|---:|---:|---:|---:|"]
    for row in contact["rows"]:
        lines.append(
            f"| {row['target_deg']:g} "
            f"| {_fmt(row['before_measured_deg'], '.2f')} | {_fmt(row['after_measured_deg'], '.2f')} "
            f"| {_fmt(row['before_error_deg'], '+.2f')} | {_fmt(row['after_error_deg'], '+.2f')} |"
        )
    lines += [
        "",
        f"- MAE: before {_fmt(contact['before_mae_deg'], '.2f')} deg, "
        f"after {_fmt(contact['after_mae_deg'], '.2f')} deg; "
        f"max |err|: before {_fmt(contact['before_max_abs_error_deg'], '.2f')}, "
        f"after {_fmt(contact['after_max_abs_error_deg'], '.2f')}",
    ]
    lines += ["", "### Flat-wall impact (recorded; no contact is required in this PR)", ""]
    lines += [
        "| case | finite b/a | beta_max b → a | y_cm final b → a | min g0.5 b → a | min g0.1 b → a "
        "| mass drift b → a | peak speed b → a |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in result["impact"]:
        if row.get("before_finite") is True and row.get("after_finite") is True:
            lines.append(
                f"| {row['case_name']} | {row['before_finite']}/{row['after_finite']} "
                f"| {row['before_beta_max']:.4f} → {row['after_beta_max']:.4f} "
                f"| {row['before_final_y_cm']:.4f} → {row['after_final_y_cm']:.4f} "
                f"| {row['before_min_g05']:.4f} → {row['after_min_g05']:.4f} "
                f"| {row['before_min_g01']:.4f} → {row['after_min_g01']:.4f} "
                f"| {row['before_mass_drift']:.2e} → {row['after_mass_drift']:.2e} "
                f"| {row['before_peak_speed']:.3f} → {row['after_peak_speed']:.3f} |"
            )
        else:
            lines.append(
                f"| {row['case_name']} | {row['before_finite']}/{row['after_finite']} "
                "| n/a | n/a | n/a | n/a | n/a | n/a |"
            )
    lines += ["", "### Blockers (before → after)", ""]
    lines += [f"- `{name}`: {b} → {a}" for name, (b, a) in result["blockers"].items()]
    lines += ["", "### Guards", ""]
    for guard in result["guards"]:
        mark = "PASS" if guard["passed"] else ("MISS" if guard["kind"] == "target" else "STOP")
        lines.append(f"- [{mark}] `{guard['name']}` ({guard['kind']}): {guard['detail']}")
    lines += [
        "",
        "**RESULT: " + ("all guards passed**" if result["guards_passed"] else "GUARD FAILED - STOP and investigate**"),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strict before/after comparison of two L1A validation reports")
    parser.add_argument("--before", required=True, help="report.json from the previous solver contract")
    parser.add_argument("--after", required=True, help="report.json from the current solver contract")
    parser.add_argument("--markdown", default=None, help="optional output path for the Markdown tables")
    parser.add_argument("--json", default=None, help="optional output path for the machine-readable comparison")
    parser.add_argument("--wet-band", type=float, default=0.15, help="solver wet_band (default 0.15) for g0.5 ratios")
    args = parser.parse_args(argv)
    try:
        reports = []
        for label, path in (("before", args.before), ("after", args.after)):
            report = json.loads(Path(path).read_text(encoding="utf-8"))
            errors = validate_report_schema(report)
            if errors:
                raise ValueError(f"{label} report {path} is not a valid L1A report: " + "; ".join(errors))
            reports.append(report)
        result = compare_reports(reports[0], reports[1], wet_band=args.wet_band)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    text = format_markdown(result)
    print(text)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(text, encoding="utf-8")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if result["guards_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
