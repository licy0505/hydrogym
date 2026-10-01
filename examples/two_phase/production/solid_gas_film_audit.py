"""One-factor-at-a-time sensitivity map of the dynamic impact no-contact gap (L1A-2b).

The L1A-2a evidence showed that the pre-contact gap of an impacting drop is
insensitive to the wetting model (``wall_energy_amp = 0`` and ``wet_band = 0.05``
leave ``min g0.5`` unchanged) while it moves with the Brinkman penalization
(``eta_pen``), the penalized-solid phase projection (``enforce_solid_phi``) and
the gas viscosity (``nu_g``).  This module exists to *quantify* that statement in
one reproducible place:

    python -m production.solid_gas_film_audit --json artifacts/solid_gas_film_audit.json

It is a diagnostic harness.  Nothing here is allowed to change production
defaults: ``eta_pen``, ``nu_g``, ``enforce_solid_phi``, the Brinkman formulation
and the solid-projection algorithm stay exactly as they are.  The only knobs the
harness turns are per-run overrides passed into
:func:`production.validation.run_impact_case`, and every run is reported with the
factor it varied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# Impact baseline shared by every audit run: flat wall, neutral wettability,
# We = 100, and the streamfunction velocity mode used by production validation.
BASE_CASE: dict[str, Any] = {
    "name": "solid_gas_film_audit",
    "We": 100.0,
    "Re": 200.0,
    "R": 0.7,
    "cos_theta": 0.0,
    "impact_gap": 0.1,
    "u_impact": 1.0,
    "velocity_mode": "streamfunction",
}

DEFAULT_FACTORS: dict[str, list[Any]] = {
    "wetting_model": ["none", "legacy_affinity", "surface_energy_volume_v6", "surface_energy"],
    "eta_pen_over_dt": [0.5, 2.0, 8.0],
    "nu_g_over_nu_l": [1.0, 10.0, 50.0],
    "phase_boundary_model": ["impermeable_flux", "projection_legacy"],
}

QUICK_FACTORS: dict[str, list[Any]] = {
    "wetting_model": ["surface_energy"],
    "eta_pen_over_dt": [2.0],
    "nu_g_over_nu_l": [10.0],
    "phase_boundary_model": ["impermeable_flux", "projection_legacy"],
}

METRIC_KEYS = (
    "minimum_gap_0.5",
    "minimum_gap_0.1",
    "first_contact_time",
    "beta_max",
    "time_to_beta_max",
    "final_y_cm",
    "peak_speed",
    "mass_drift",
    "solid_phase_fraction",
    "implicit_iterations_max",
    "implicit_relative_residual_max",
    "finite",
)


def _case_with_overrides(factor: str, value: Any) -> dict[str, Any]:
    """Return a copy of the baseline case with exactly one factor overridden."""
    case = dict(BASE_CASE)
    if factor == "wetting_model":
        case["wetting_model"] = value
    elif factor == "eta_pen_over_dt":
        case["eta_pen_over_dt"] = float(value)
    elif factor == "nu_g_over_nu_l":
        case["nu_g_over_nu_l"] = float(value)
    elif factor == "phase_boundary_model":
        case["phase_boundary_model"] = str(value)
        # The named legacy model is the archived v6 projection path. The new
        # impermeable-flux path must never call the post-step redistribution.
        case["enforce_solid_phi"] = value == "projection_legacy"
    else:
        raise ValueError(f"unknown audit factor {factor!r}")
    return case


def _metrics(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("finite") is not True:
        return {key: None for key in METRIC_KEYS} | {"error": record.get("error", "unknown failure")}
    series = record.get("time_series", {})
    speeds = [float(v) for v in series.get("max_speed", [])]
    return {
        "minimum_gap_0.5": record.get("minimum_gap_0.5"),
        "minimum_gap_0.1": float(min(series.get("g_0.1", [math.inf]))),
        "first_contact_time": record.get("first_contact_time"),
        "beta_max": record.get("beta_max"),
        "time_to_beta_max": record.get("time_to_beta_max"),
        "final_y_cm": record.get("final_y_cm"),
        "peak_speed": float(max(speeds)) if speeds else None,
        "mass_drift": record.get("mass_drift"),
        "solid_phase_fraction": record.get("solid_phase_fraction"),
        "implicit_iterations_max": record.get("implicit_iterations_max"),
        "implicit_relative_residual_max": record.get("implicit_relative_residual_max"),
        "finite": True,
    }


def run_solid_gas_film_audit(
    *,
    N: int = 128,
    steps: int = 1200,
    save_every: int = 20,
    dt: float = 2.0e-3,
    eps_factor: float = 2.0,
    dtype: str = "float32",
    factors: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Run the one-factor-at-a-time impact ablation and return a JSON-safe summary."""
    from production.validation import run_impact_case

    grid = DEFAULT_FACTORS if factors is None else factors
    settings = {
        "N": int(N),
        "steps": int(steps),
        "save_every": int(save_every),
        "dt": float(dt),
        "eps_factor": float(eps_factor),
        "dtype": dtype,
        "baseline_case": dict(BASE_CASE),
        "diagnostic_only": True,
        "production_defaults_changed": False,
    }
    runs: list[dict[str, Any]] = []
    for factor in ("wetting_model", "eta_pen_over_dt", "nu_g_over_nu_l", "phase_boundary_model"):
        for value in grid.get(factor, []):
            case = _case_with_overrides(factor, value)
            record = run_impact_case(
                case,
                N=N,
                eps_factor=eps_factor,
                steps=steps,
                save_every=save_every,
                dt=dt,
                dtype=dtype,
            )
            runs.append(
                {
                    "factor": factor,
                    "value": value,
                    "wetting_model": case.get("wetting_model", "config_default"),
                    "case_parameters": {
                        "eta_pen_over_dt": case.get("eta_pen_over_dt"),
                        "nu_g_over_nu_l": case.get("nu_g_over_nu_l"),
                        "phase_boundary_model": case.get("phase_boundary_model", "impermeable_flux"),
                        "enforce_solid_phi": case.get("enforce_solid_phi", False),
                    },
                    "metrics": _metrics(record),
                    "finite": bool(record.get("finite") is True),
                }
            )
    return {
        "audit_schema_version": 1,
        "audit": "solid_gas_film",
        "settings": settings,
        "runs": runs,
        "notes": [
            "Diagnostic only: no production default (eta_pen, nu_g, enforce_solid_phi, Brinkman, "
            "solid projection) is modified by this module.",
            "A remaining finite gap after this audit is an L1A-2c (solid boundary / gas film) question, "
            "not a wetting-model failure.",
        ],
    }


def format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Solid / gas-film impact ablation (diagnostic only)",
        "",
        f"- N = {result['settings']['N']}, steps = {result['settings']['steps']}, "
        f"dt = {result['settings']['dt']}, eps/dx = {result['settings']['eps_factor']}",
        f"- baseline case: `{json.dumps(result['settings']['baseline_case'], sort_keys=True)}`",
        "",
        "| factor | value | min g0.5 | min g0.1 | first contact t | beta max | final y_cm | peak speed | mass drift |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in result["runs"]:
        metrics = run["metrics"]
        if not run["finite"]:
            lines.append(f"| {run['factor']} | {run['value']} | FAILED | | | | | | {metrics.get('error', '')} |")
            continue
        lines.append(
            (
                "| {factor} | {value} | {g05:.4f} | {g01:.4f} | {contact} | "
                "{beta:.4f} | {y:.4f} | {speed:.4g} | {mass:.3e} |"
            ).format(
                factor=run["factor"],
                value=run["value"],
                g05=metrics["minimum_gap_0.5"],
                g01=metrics["minimum_gap_0.1"],
                contact="-" if metrics["first_contact_time"] is None else f"{metrics['first_contact_time']:.3f}",
                beta=metrics["beta_max"],
                y=metrics["final_y_cm"],
                speed=metrics["peak_speed"],
                mass=metrics["mass_drift"],
            )
        )
    lines.append("")
    lines.append("Diagnostic only: no production default is tuned by this table.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Impact no-contact-gap sensitivity audit (diagnostic only)")
    parser.add_argument("--json", type=str, default=None, help="write the machine-readable result here")
    parser.add_argument("--markdown", type=str, default=None, help="write a markdown table here")
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--dt", type=float, default=2.0e-3)
    parser.add_argument("--eps-factor", type=float, default=2.0)
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--quick", action="store_true", help="run only the baseline point of every factor")
    args = parser.parse_args(argv)

    result = run_solid_gas_film_audit(
        N=args.N,
        steps=args.steps,
        save_every=args.save_every,
        dt=args.dt,
        eps_factor=args.eps_factor,
        dtype=args.dtype,
        factors=QUICK_FACTORS if args.quick else None,
    )
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    markdown = format_markdown(result)
    if args.markdown:
        path = Path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
