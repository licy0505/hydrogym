"""Contact-angle comparison matrix: historical v5 (A), clean legacy v5 (B), v7 (C).

The two model generations can only be compared on the *same* measurement and the
*same* initial state:

* **A - historical v5**: the L1A-2a baseline.  Overlapping seed
  (``y0 = wall + R - min(0.15, 0.14 R)``), legacy volumetric wall affinity and the
  legacy area/width measurement.  The recorded numbers are kept in
  :data:`HISTORICAL_V5`; ``--with-historical`` re-runs the pipeline for a
  reproducibility check.
* **B - clean v5**: ``wetting_model='legacy_affinity'`` on the clean sessile seed
  with the contour measurement.  This is the model-comparison baseline: only the
  wall model changes between B and C.
* **C - clean v7**: the shipped ``surface_energy`` natural-BC default on the clean seed.

Only converged runs (``windows`` consecutive samples within ``angle_tol_deg`` and
``speed_tol``) contribute to MAE/RMSE/max error; the matrix reports the converged
count and the last sampled angle for everything else.  A drifting angle is not an
equilibrium contact angle.

This module is diagnostic tooling: it never edits the solver and never feeds a
result back into the solver.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np
import phasefield as pf
from production import observables as obs
from production.validation import _params, run_contact_angle_case

#: L1A-2a recorded baseline (contract v5, overlapping seed, legacy measurement).
HISTORICAL_V5 = {
    60.0: 119.78,
    90.0: 119.96,
    120.0: 120.12,
    150.0: 117.32,
}

DEFAULT_TARGETS = (60.0, 90.0, 120.0, 150.0)


def run_historical_case(
    target_deg: float,
    N: int = 128,
    steps: int = 3000,
    R: float = 1.1,
    dt: float = 4.0e-3,
    eps_factor: float = 2.0,
    wall_energy_amp: float = 5.0,
) -> dict[str, Any]:
    """Re-run the contract-v5 pipeline (overlapping seed + legacy measurement)."""
    wall_height = 0.25
    p = _params(
        N,
        Re=200.0,
        We=100.0,
        eps_factor=eps_factor,
        dt=dt,
        dtype="float32",
        wall_energy_amp=wall_energy_amp,
        enforce_solid_phi=False,
        wetting_model="legacy_affinity",
        phase_boundary_model="projection_legacy",
    )
    solid = pf.make_solid(pf.surface_flat(p, wall_height=wall_height), p, cos_theta=math.cos(math.radians(target_deg)))
    y0 = wall_height + R - min(0.15, 0.14 * R)
    state = pf.droplet_initial_state(p, x0=p.Lx / 2.0, y0=y0, R=R, u_impact=0.0)
    mass_initial = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
    step_fn = jax.jit(pf.step, static_argnums=(2,))
    started = time.perf_counter()
    for _ in range(int(steps)):
        state = step_fn(state, solid, p)
    elapsed = time.perf_counter() - started
    measured = float(pf.measure_contact_angle_area_width(state.phi, solid, p))
    mass_final = obs.liquid_mass(state.phi, solid.sdf, p.dx, p.dy)
    return {
        "target_deg": float(target_deg),
        "measured_deg": measured,
        "absolute_error_deg": abs(measured - float(target_deg)),
        "converged": None,
        "relaxation_steps": int(steps),
        "mass_relative_drift": abs(mass_final - mass_initial) / max(abs(mass_initial), 1e-12),
        "initial_solid_liquid_fraction": None,
        "runtime": {"steps": int(steps), "wall_seconds": float(elapsed)},
        "note": "contract-v5 pipeline: overlapping seed, legacy measurement (not an equilibrium criterion)",
    }


def _profile_rows(
    name: str,
    targets: list[float],
    *,
    N: int,
    max_steps: int,
    dt: float,
    eps_factor: float,
    R: float,
    sample_every: int,
    angle_tol_deg: float,
    speed_tol: float,
    windows: int,
) -> dict[str, Any]:
    rows = []
    for target in targets:
        if name == "history":
            rows.append(run_historical_case(target, N=N, steps=max_steps, R=R, dt=dt, eps_factor=eps_factor))
            continue
        case = run_contact_angle_case(
            target,
            N=N,
            max_steps=max_steps,
            R=R,
            dt=dt,
            eps_factor=eps_factor,
            dtype="float32",
            wetting_model="legacy_affinity" if name == "legacy_clean" else "surface_energy",
            phase_boundary_model="projection_legacy" if name == "legacy_clean" else "impermeable_flux",
            enforce_solid_phi=name == "legacy_clean",
            sample_every=sample_every,
            angle_tol_deg=angle_tol_deg,
            speed_tol=speed_tol,
            windows=windows,
        )
        rows.append(case.to_dict())
        print(
            f"  {name:15s} target={target:6.1f} converged={case.converged} "
            f"final={case.final_sampled_deg if case.final_sampled_deg is None else round(case.final_sampled_deg, 3)} "
            f"steps={case.relaxation_steps} dMf={case.mass_relative_drift:.2e}",
            flush=True,
        )
    return {"profile": name, "cases": rows, "summary": summarize_rows(rows)}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize equilibrium errors without treating a partial matrix as complete.

    Historical contract-v5 values remain a separately identified recorded matrix.
    For equilibrium profiles, only converged cases have ``measured_deg``; when only
    a subset converges, overall MAE/RMSE/max remain ``None`` and the subset MAE is
    reported separately as a diagnostic.
    """
    ordered = sorted(rows, key=lambda row: float(row["target_deg"]))
    reported = [
        row
        for row in ordered
        if isinstance(row.get("measured_deg"), (int, float)) and math.isfinite(float(row["measured_deg"]))
    ]
    converged = [row for row in ordered if row.get("converged") is True]
    converged_reported = [
        row
        for row in reported
        if row.get("converged") is True
        and isinstance(row.get("absolute_error_deg"), (int, float))
        and math.isfinite(float(row["absolute_error_deg"]))
    ]
    reported_errors = np.asarray([float(row["absolute_error_deg"]) for row in reported], dtype=np.float64)
    converged_errors = np.asarray([float(row["absolute_error_deg"]) for row in converged_reported], dtype=np.float64)
    angles = [float(row["measured_deg"]) for row in reported]
    historical_complete = (
        bool(ordered) and len(reported) == len(ordered) and all(row.get("converged") is None for row in ordered)
    )
    equilibrium_complete = bool(ordered) and len(converged_reported) == len(ordered)
    reportable = historical_complete or equilibrium_complete
    errors = reported_errors if historical_complete else (converged_errors if equilibrium_complete else np.asarray([]))
    monotonic = all(b >= a for a, b in zip(angles, angles[1:]))
    return {
        "case_count": len(ordered),
        "reported_case_count": len(reported),
        "converged_case_count": len(converged),
        "converged_targets": [float(row["target_deg"]) for row in converged],
        "all_targets_converged": bool(equilibrium_complete),
        "converged_subset_mae_deg": float(converged_errors.mean()) if converged_errors.size else None,
        "mae_deg": float(errors.mean()) if errors.size else None,
        "rmse_deg": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
        "max_absolute_error_deg": float(errors.max()) if errors.size else None,
        "monotonic_target_to_measured": bool(monotonic) if reportable and len(reported) == len(ordered) else False,
        "angles_deg": [
            {
                "target_deg": float(row["target_deg"]),
                "measured_deg": row.get("measured_deg"),
                "final_sampled_deg": row.get("final_sampled_deg"),
                "converged": row.get("converged"),
            }
            for row in ordered
        ],
    }


def format_matrix(matrix: dict[str, Any]) -> str:
    lines = [
        (
            "| profile | target (deg) | measured / final (deg) | converged | MAE (deg) | "
            "RMSE (deg) | max err (deg) | monotonic |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, block in matrix["profiles"].items():
        summary = block["summary"]
        mae = "n/a" if summary["mae_deg"] is None else f"{summary['mae_deg']:.3f}"
        rmse = "n/a" if summary["rmse_deg"] is None else f"{summary['rmse_deg']:.3f}"
        max_err = "n/a" if summary["max_absolute_error_deg"] is None else f"{summary['max_absolute_error_deg']:.3f}"
        for row in summary["angles_deg"]:
            value = row["measured_deg"] if row["converged"] is not False else row["final_sampled_deg"]
            value = "n/a" if value is None else f"{float(value):.3f}"
            lines.append(
                f"| {name} | {row['target_deg']:.0f} | {value} | {row['converged']} | {mae} | {rmse} | {max_err} | "
                f"{summary['monotonic_target_to_measured']} |"
            )
    return "\n".join(lines)


def build_matrix(
    targets: list[float],
    profiles: list[str],
    *,
    N: int = 128,
    max_steps: int = 8000,
    dt: float = 4.0e-3,
    eps_factor: float = 2.0,
    R: float = 1.1,
    sample_every: int = 200,
    angle_tol_deg: float = 0.25,
    speed_tol: float = 5e-4,
    windows: int = 3,
) -> dict[str, Any]:
    matrix: dict[str, Any] = {
        "targets": [float(t) for t in targets],
        "settings": {
            "N": int(N),
            "max_steps": int(max_steps),
            "dt": float(dt),
            "eps_factor": float(eps_factor),
            "R": float(R),
            "sample_every": int(sample_every),
            "angle_tol_deg": float(angle_tol_deg),
            "speed_tol": float(speed_tol),
            "windows": int(windows),
        },
        "profiles": {},
    }
    for name in profiles:
        if name == "history":
            print("  re-running the contract-v5 pipeline for the reproducibility check", flush=True)
        matrix["profiles"][name] = _profile_rows(
            name,
            targets,
            N=N,
            max_steps=max_steps,
            dt=dt,
            eps_factor=eps_factor,
            R=R,
            sample_every=sample_every,
            angle_tol_deg=angle_tol_deg,
            speed_tol=speed_tol,
            windows=windows,
        )
    if "history" not in profiles:
        matrix["profiles"]["history_recorded"] = {
            "profile": "history_recorded",
            "cases": [],
            "summary": summarize_rows(
                [
                    {
                        "target_deg": target,
                        "measured_deg": value,
                        "absolute_error_deg": abs(value - target),
                        "converged": None,
                    }
                    for target, value in sorted(HISTORICAL_V5.items())
                ]
            ),
        }
    return matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS))
    parser.add_argument("--profiles", nargs="+", default=["legacy_clean", "surface_energy"])
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--dt", type=float, default=4.0e-3)
    parser.add_argument("--eps-factor", type=float, default=2.0)
    parser.add_argument("--R", type=float, default=1.1)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--angle-tol-deg", type=float, default=0.25)
    parser.add_argument("--speed-tol", type=float, default=5e-4)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--with-historical", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    profiles = list(args.profiles)
    if args.with_historical and "history" not in profiles:
        profiles.append("history")
    max_steps = 600 if args.quick else args.max_steps
    matrix = build_matrix(
        args.targets,
        profiles,
        N=args.N,
        max_steps=max_steps,
        dt=args.dt,
        eps_factor=args.eps_factor,
        R=args.R,
        sample_every=20 if args.quick else args.sample_every,
        angle_tol_deg=args.angle_tol_deg,
        speed_tol=args.speed_tol,
        windows=args.windows,
    )
    print(format_matrix(matrix))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(matrix, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
