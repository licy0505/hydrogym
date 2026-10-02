"""Generate validated droplet-impact trajectories for the two-phase surrogate.

The old generator wrote every solver trajectory to disk.  That allowed three
silent problems into the training set:

* the default drop centre overlapped the wall;
* the saved frame spacing did not match the physical time advanced by
  ``phasefield.step``;
* NaN-free but nonphysical trajectories (mass loss, liquid in the solid, or
  phase overshoot) were accepted as labels.

This script keeps the existing ``.npz`` schema, but makes the simulation
contract explicit: build a non-overlapping case, integrate for the requested
physical time, validate the complete trajectory, and only then write it.

Examples
--------

.. code-block:: bash

    python generate_dataset.py --set lowWe --out data/lowWe --nsteps 2000
    python generate_dataset.py --set base --out data/base --nsteps 2000 --ds 3
    python generate_dataset.py --set lowWe --out data/lowWe --limit 2 --dry-run

``--nsteps`` and ``--save_every`` are interpreted using ``--dt`` as the
requested physical time scale.  Case-specific ``dt`` values are respected and
the effective values are saved in the per-case metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cases as C
import jax
import numpy as np
import phasefield as pf
from scipy.ndimage import distance_transform_edt

DATASET_SCHEMA_VERSION = 3


def _case_name(case: dict, set_name: str, index: int) -> str:
    """Return a stable, filesystem-safe case name."""
    if set_name == "base":
        label = C.case_label({k: v for k, v in case.items() if k != "family"})
        return f"{case['split']}_{index:02d}_{label}"
    return f"{case['split']}_{set_name}_{index:03d}_{case['surface']}"


def _source_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dataset_fingerprint(case, args, dt, nsteps, save_every) -> str:
    """Hash every input that can change the generated trajectory or saved grid."""
    payload = dict(
        schema=DATASET_SCHEMA_VERSION,
        solver_contract=int(pf.SOLVER_CONTRACT_VERSION),
        wetting_model=str(case.get("wetting_model", "surface_energy")),
        phase_boundary_model=str(case.get("phase_boundary_model", "impermeable_flux")),
        wall_measure=str(pf.WALL_MEASURE_METHOD),
        wall_measure_contract_version=int(pf.WALL_MEASURE_CONTRACT_VERSION),
        solver_sha256=_source_sha256(pf.__file__),
        generator_sha256=_source_sha256(__file__),
        case=case,
        N=int(args.N),
        ds=int(args.ds),
        dt=float(dt),
        nsteps=int(nsteps),
        save_every=int(save_every),
        validation=dict(
            max_phi_overshoot=float(getattr(args, "max_phi_overshoot", 0.02)),
            max_solid_leak=float(getattr(args, "max_solid_leak", 5e-4)),
            min_total_mass_ratio=float(getattr(args, "min_total_mass_ratio", 0.995)),
            max_total_mass_ratio=float(getattr(args, "max_total_mass_ratio", 1.005)),
            max_speed=float(getattr(args, "max_speed", 5.0)),
            min_feature_cells=float(getattr(args, "min_feature_cells", 2.0)),
        ),
    )
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _saved_case_is_current(path: Path, expected_fingerprint: str) -> bool:
    """Return True only when schema and exact generation fingerprint match."""
    try:
        with np.load(path, allow_pickle=True) as d:
            if "dataset_schema_version" not in d.files or "dataset_fingerprint" not in d.files:
                return False
            version = int(np.asarray(d["dataset_schema_version"]).item())
            fingerprint = str(np.asarray(d["dataset_fingerprint"]).item())
            return version == DATASET_SCHEMA_VERSION and fingerprint == expected_fingerprint
    except Exception:
        return False


def _smallest_feature_size(case: dict) -> float:
    surface = case.get("surface", "flat")
    if surface == "pillars":
        return float(case.get("width", 0.3))
    if surface == "random_pillars":
        return float(case.get("width_range", (0.15, 0.4))[0])
    if surface == "hierarchical":
        return float(min(case.get("width", 0.6), case.get("sub_width", 0.1)))
    if surface == "grooves":
        return float(case.get("width", 0.25))
    return float("inf")


def _feature_cells(case: dict, saved_dx: float) -> float:
    size = _smallest_feature_size(case)
    return float("inf") if not np.isfinite(size) else size / saved_dx


def _effective_schedule(case: dict, args: argparse.Namespace) -> tuple[float, int, int]:
    """Resolve the actual solver dt and integer save schedule."""
    requested_dt = float(case.get("dt", args.dt))
    p0 = pf.PhaseFieldParams(Nx=args.N, Ny=args.N, Lx=6.0, Ly=6.0, dt=requested_dt)
    dt_cap = float(pf.stable_dt(p0, u_max=2.0))
    dt = min(requested_dt, dt_cap)
    horizon = max(float(args.nsteps) * float(args.dt), dt)
    save_time = max(float(args.save_every) * float(args.dt), dt)
    save_every = max(1, int(round(save_time / dt)))
    nsteps = max(save_every, int(round(horizon / dt)))
    nsteps = max(save_every, (nsteps // save_every) * save_every)
    return dt, nsteps, save_every


def _downsample_history(history: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool a ``(T,N,N)`` history without changing its mass density."""
    if factor == 1:
        return history
    T, Nx, Ny = history.shape
    if Nx % factor or Ny % factor:
        raise ValueError(f"downsample factor {factor} does not divide ({Nx}, {Ny})")
    return history.reshape(T, Nx // factor, factor, Ny // factor, factor).mean(axis=(2, 4))


def _coarsen_geometry(solid: pf.Solid, factor: int, fine_dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Return coarse solid fraction and a reinitialised coarse signed distance."""
    hard = (np.asarray(solid.sdf) < 0.0).astype(np.float32)
    frac = _downsample_history(hard[None], factor)[0].astype(np.float32)
    if factor == 1:
        return frac, np.asarray(solid.sdf, dtype=np.float32)

    mask = frac >= 0.5
    saved_dx = fine_dx * factor
    # Geometry is periodic in x. Tile only the x-axis before the EDT so points
    # near the left/right seam see the correct nearest solid across the seam.
    tiled = np.concatenate([mask, mask, mask], axis=0)
    outside_tiled = distance_transform_edt(~tiled) * saved_dx
    inside_tiled = distance_transform_edt(tiled) * saved_dx
    nx = mask.shape[0]
    outside = outside_tiled[nx : 2 * nx]
    inside = inside_tiled[nx : 2 * nx]
    sdf = (outside - inside).astype(np.float32)
    return frac, sdf


def _diagnose(
    initial: pf.State,
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    solid: pf.Solid,
    p: pf.PhaseFieldParams,
    *,
    max_phi_overshoot: float,
    max_solid_leak: float,
    min_total_mass_ratio: float,
    max_total_mass_ratio: float,
    max_speed: float,
) -> tuple[bool, dict]:
    """Validate finite values, conservative mass and *deep-solid* leakage."""
    # Histories begin at the first saved frame (after ``save_every`` steps).
    # The hard-fluid initialization clips only the tiny diffuse tail in solid at
    # t=0; from then on both total and fluid-region mass are audited against that
    # same unprojected initial condition. No startup redistribution is expected in v7.
    raw_initial = np.asarray(initial.phi, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    first_saved = np.asarray(phi[0] if phi.shape[0] else raw_initial, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    hard_solid = np.asarray(solid.sdf < 0.0, dtype=np.float64)
    fluid = 1.0 - hard_solid

    raw_total0 = float(np.sum(raw_initial))
    first_saved_total = float(np.sum(first_saved))
    total = np.sum(phi, axis=(1, 2))
    raw_initial_fluid0 = float(np.sum(raw_initial * fluid))
    fluid_mass = np.sum(phi * fluid[None], axis=(1, 2))
    denom = np.maximum(np.sum(np.abs(phi), axis=(1, 2)), 1e-12)
    raw_initial_denom = max(float(np.sum(np.abs(raw_initial))), 1e-12)
    leak = np.sum(np.abs(phi) * hard_solid[None], axis=(1, 2)) / denom
    raw_initial_solid_leak = float(np.sum(np.abs(raw_initial) * hard_solid) / raw_initial_denom)
    speed = np.sqrt(u * u + v * v)

    finite = bool(
        np.isfinite(raw_initial).all() and np.isfinite(phi).all() and np.isfinite(u).all() and np.isfinite(v).all()
    )
    overshoot = float(max(np.max(-phi), np.max(phi - 1.0), 0.0)) if phi.size else 0.0
    total_ratio = total / max(raw_total0, 1e-12)
    fluid_ratio = fluid_mass / max(raw_initial_fluid0, 1e-12)
    startup_total_mass_ratio = first_saved_total / max(raw_total0, 1e-12)
    startup_fluid_mass_ratio = fluid_mass[0] / max(raw_initial_fluid0, 1e-12)
    diag = {
        "finite": finite,
        "initial_total_mass": raw_total0 * p.dx * p.dy,
        "first_saved_total_mass": first_saved_total * p.dx * p.dy,
        "startup_total_mass_ratio": float(startup_total_mass_ratio),
        "startup_fluid_mass_ratio": float(startup_fluid_mass_ratio),
        "raw_initial_fluid_mass": raw_initial_fluid0 * p.dx * p.dy,
        "raw_initial_solid_leak": raw_initial_solid_leak,
        "final_total_mass_ratio": float(total_ratio[-1]),
        "min_total_mass_ratio": float(np.min(total_ratio)),
        "max_total_mass_ratio": float(np.max(total_ratio)),
        "final_fluid_mass_ratio": float(fluid_ratio[-1]),
        "min_fluid_mass_ratio": float(np.min(fluid_ratio)),
        "max_fluid_mass_ratio": float(np.max(fluid_ratio)),
        "initial_solid_leak": float(leak[0]),
        "max_solid_leak": float(np.max(leak)),
        "max_phi_overshoot": overshoot,
        "max_speed": float(np.max(speed)),
        "shape": list(phi.shape),
    }
    ok = (
        finite
        and diag["min_total_mass_ratio"] >= min_total_mass_ratio
        and diag["max_total_mass_ratio"] <= max_total_mass_ratio
        and diag["min_fluid_mass_ratio"] >= min_total_mass_ratio
        and diag["max_fluid_mass_ratio"] <= max_total_mass_ratio
        and diag["max_solid_leak"] <= max_solid_leak
        and diag["max_phi_overshoot"] <= max_phi_overshoot
        and diag["max_speed"] <= max_speed
    )
    return bool(ok), diag


def _rejection_path(out_dir: Path, name: str) -> Path:
    return out_dir / "rejected" / f"{name}.json"


def _remove_rejection(out_dir: Path, name: str) -> None:
    _rejection_path(out_dir, name).unlink(missing_ok=True)


def _write_rejection(out_dir: Path, name: str, case: dict, diagnostics: dict) -> None:
    reject_dir = out_dir / "rejected"
    reject_dir.mkdir(parents=True, exist_ok=True)
    payload = {"case": case, "diagnostics": diagnostics}
    _rejection_path(out_dir, name).write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _manifest_record(name: str, case: dict, status: str, fingerprint: str | None = None) -> dict:
    record = {
        "case_name": name,
        "split": case["split"],
        "surface": case.get("surface", "flat"),
        "status": status,
    }
    if fingerprint is not None:
        record["trajectory_fingerprint"] = fingerprint
    return record


def _write_manifest(out_dir: Path, set_name: str, records: list[dict]) -> dict:
    accepted_statuses = {"current", "generated"}
    accepted = [r for r in records if r["status"] in accepted_statuses]
    rejected = [
        r for r in records if r["status"] in {"underresolved_geometry", "physics_validation_rejection", "exception"}
    ]
    fingerprints = sorted(str(r["trajectory_fingerprint"]) for r in accepted if r.get("trajectory_fingerprint"))
    aggregate = hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()
    accepted_by_surface = {}
    for record in accepted:
        surface = record["surface"]
        accepted_by_surface[surface] = accepted_by_surface.get(surface, 0) + 1
    manifest = {
        "manifest_schema_version": 1,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
        "wetting_model": "surface_energy",
        "phase_boundary_model": "impermeable_flux",
        "wall_measure_method": str(pf.WALL_MEASURE_METHOD),
        "wall_measure_contract_version": int(pf.WALL_MEASURE_CONTRACT_VERSION),
        "case_set": set_name,
        "expected": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "complete": len(records) == len(accepted) and not rejected,
        "accepted_by_surface": accepted_by_surface,
        "aggregate_fingerprint": aggregate,
        "records": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def _save_case(
    path: Path,
    case: dict,
    p: pf.PhaseFieldParams,
    solid: pf.Solid,
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    save_every: int,
    ds: int,
    diagnostics: dict,
    dataset_fingerprint: str,
    feature_cells_min: float,
    parameter_semantics: dict,
) -> None:
    """Write one validated trajectory with explicit physical-time metadata."""
    # Keep phi/geometry in float32: thin low-We films can sit near the plotting
    # threshold and should not lose additional information to float16 quantisation.
    phi_d = _downsample_history(phi, ds).astype(np.float32)
    u_d = _downsample_history(u, ds).astype(np.float16)
    v_d = _downsample_history(v, ds).astype(np.float16)
    chi_d, sdf_d = _coarsen_geometry(solid, ds, p.dx)
    sdf_d = np.clip(sdf_d, -1.0, 3.0).astype(np.float32)
    saved_dx = float(p.dx * ds)
    scalars = np.array(
        [case.get("We", 100.0) / 100.0, case.get("Re", 200.0) / 200.0, float(case.get("cos_theta", 0.0)), saved_dx],
        dtype=np.float32,
    )
    case_meta = dict(case)
    case_meta.update(
        dataset_schema_version=DATASET_SCHEMA_VERSION,
        solver_contract_version=int(pf.SOLVER_CONTRACT_VERSION),
        wetting_model=str(p.wetting_model),
        phase_boundary_model=str(p.phase_boundary_model),
        wall_measure_method=str(p.wall_measure),
        wall_measure_contract_version=int(pf.WALL_MEASURE_CONTRACT_VERSION),
        cos_theta_semantics=(
            "target Young equilibrium contact-angle cosine (surface_energy) or legacy wall-affinity cosine"
        ),
        solver_dt=float(p.dt),
        solver_dx=float(p.dx),
        saved_dx=saved_dx,
        save_every=int(save_every),
        frame_dt=float(p.dt * save_every),
        feature_cells_min=float(feature_cells_min),
        dataset_fingerprint=dataset_fingerprint,
        solver_sha256=_source_sha256(pf.__file__),
        parameter_semantics=parameter_semantics,
        diagnostics=diagnostics,
    )
    times = np.arange(1, phi_d.shape[0] + 1, dtype=np.float32) * float(p.dt * save_every)
    np.savez_compressed(
        path,
        phi=phi_d,
        u=u_d,
        v=v_d,
        chi=chi_d,
        sdf=sdf_d,
        scalars=scalars,
        time=times,
        dataset_schema_version=np.array(DATASET_SCHEMA_VERSION, dtype=np.int32),
        dataset_fingerprint=np.array(dataset_fingerprint),
        solver_sha256=np.array(_source_sha256(pf.__file__)),
        feature_cells_min=np.array(feature_cells_min, dtype=np.float32),
        surface=np.array(case.get("surface", "flat")),
        split=np.array(case["split"]),
        case=np.array(json.dumps(case_meta, ensure_ascii=False)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="base", choices=sorted(C.CASE_SETS))
    ap.add_argument("--out", default="data")
    ap.add_argument("--nsteps", type=int, default=2000, help="nominal steps at --dt; defines physical horizon")
    ap.add_argument("--save_every", type=int, default=20, help="nominal steps at --dt between saved frames")
    ap.add_argument("--ds", type=int, default=3, help="average-pooling factor (N/ds is saved resolution)")
    ap.add_argument("--N", type=int, default=192, help="solver resolution")
    ap.add_argument("--dt", type=float, default=4e-3, help="nominal solver timestep")
    ap.add_argument("--limit", type=int, default=0, help="max cases (0 = all)")
    ap.add_argument("--overwrite", action="store_true", help="regenerate existing .npz files")
    ap.add_argument("--require-complete", action="store_true", help="fail if any planned case is not accepted")
    ap.add_argument("--dry-run", action="store_true", help="build cases and print schedule without integrating")
    ap.add_argument("--max-phi-overshoot", type=float, default=0.02)
    ap.add_argument("--max-solid-leak", type=float, default=5e-4)
    ap.add_argument("--min-total-mass-ratio", type=float, default=0.995)
    ap.add_argument("--max-total-mass-ratio", type=float, default=1.005)
    ap.add_argument("--max-speed", type=float, default=5.0)
    ap.add_argument(
        "--min-feature-cells",
        type=float,
        default=2.0,
        help="reject discrete micro-features narrower than this many saved-grid cells",
    )
    args = ap.parse_args()

    if args.N % args.ds:
        ap.error(f"--ds={args.ds} must divide --N={args.N}")
    if args.nsteps <= 0 or args.save_every <= 0:
        ap.error("--nsteps and --save_every must be positive")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_cases = C.CASE_SETS[args.set]()
    if args.limit:
        all_cases = all_cases[: args.limit]

    records = []
    for i, case in enumerate(all_cases):
        name = _case_name(case, args.set, i)
        path = out_dir / f"{name}.npz"
        temporary_path = out_dir / f".{name}.tmp.npz"
        _remove_rejection(out_dir, name)

        dt, nsteps, save_every = _effective_schedule(case, args)
        fingerprint = _dataset_fingerprint(case, args, dt, nsteps, save_every)
        saved_dx = 6.0 / args.N * args.ds
        feature_cells_min = _feature_cells(case, saved_dx)
        semantics = {
            "nominal_We": float(case.get("We", 100.0)),
            "nominal_Re": float(case.get("Re", 200.0)),
            "u_impact_star": float(case.get("u_impact", 0.5)),
            "kinematic_We": float(case.get("We", 100.0)) * float(case.get("u_impact", 0.5)) ** 2,
            "kinematic_Re": float(case.get("Re", 200.0)) * abs(float(case.get("u_impact", 0.5))),
            "solver_contract_version": int(pf.SOLVER_CONTRACT_VERSION),
            # The case pipeline records the exact production defaults; each
            # trajectory file also carries the instantiated parameters.
            "wetting_model": str(case.get("wetting_model", "surface_energy")),
            "phase_boundary_model": str(case.get("phase_boundary_model", "impermeable_flux")),
            "wall_measure_method": str(pf.WALL_MEASURE_METHOD),
            "wall_measure_contract_version": int(pf.WALL_MEASURE_CONTRACT_VERSION),
            "cos_theta": (
                "target Young equilibrium contact-angle cosine"
                if str(case.get("wetting_model", "surface_energy")) in {"surface_energy", "surface_energy_volume_v6"}
                else "legacy wall-affinity cosine (contract <= 5)"
            ),
        }

        if path.exists() and not args.overwrite and _saved_case_is_current(path, fingerprint):
            print(f"[{i:03d}] exact fingerprint match, skip: {path.name}", flush=True)
            records.append(_manifest_record(name, case, "current", fingerprint))
            continue

        if path.exists():
            reason = "overwrite" if args.overwrite else "stale fingerprint"
            print(f"[{i:03d}] {reason}, remove and regenerate: {path.name}", flush=True)
            path.unlink(missing_ok=True)
        temporary_path.unlink(missing_ok=True)

        print(
            f"[{i:03d}] {name}: dt={dt:g}, nsteps={nsteps}, save_every={save_every}, "
            f"feature_cells={feature_cells_min:.2f}",
            flush=True,
        )
        if feature_cells_min < args.min_feature_cells:
            diagnostics = {
                "reason": "underresolved_geometry",
                "feature_cells_min": feature_cells_min,
                "required_feature_cells": args.min_feature_cells,
                "saved_dx": saved_dx,
            }
            _write_rejection(out_dir, name, case, diagnostics)
            record = _manifest_record(name, case, "underresolved_geometry")
            record["diagnostics"] = diagnostics
            records.append(record)
            print(f"[{i:03d}] REJECT underresolved geometry: {diagnostics}", flush=True)
            continue
        if args.dry_run:
            records.append(_manifest_record(name, case, "dry_run"))
            print(f"[{i:03d}] dry-run: no trajectory generated", flush=True)
            continue

        t0 = time.time()
        try:
            p, solid, initial = pf.build_case(case, N=args.N, dt=dt)
            final, phi, u, v = pf.rollout(initial, solid, p, nsteps, save_every=save_every)
            del final
            ok, diagnostics = _diagnose(
                initial,
                np.asarray(phi),
                np.asarray(u),
                np.asarray(v),
                solid,
                p,
                max_phi_overshoot=args.max_phi_overshoot,
                max_solid_leak=args.max_solid_leak,
                min_total_mass_ratio=args.min_total_mass_ratio,
                max_total_mass_ratio=args.max_total_mass_ratio,
                max_speed=args.max_speed,
            )
            if not ok:
                _write_rejection(out_dir, name, case, diagnostics)
                record = _manifest_record(name, case, "physics_validation_rejection")
                record["diagnostics"] = diagnostics
                records.append(record)
                print(f"[{i:03d}] REJECT physics validation: {diagnostics}", flush=True)
                continue
            _save_case(
                temporary_path,
                case,
                p,
                solid,
                np.asarray(phi),
                np.asarray(u),
                np.asarray(v),
                save_every,
                args.ds,
                diagnostics,
                fingerprint,
                feature_cells_min,
                semantics,
            )
            temporary_path.replace(path)
            _remove_rejection(out_dir, name)
            records.append(_manifest_record(name, case, "generated", fingerprint))
            print(f"[{i:03d}] saved T={phi.shape[0]} in {time.time() - t0:.1f}s -> {path}", flush=True)
        except Exception as exc:
            # The destination was removed before integration and writes are
            # atomic, so a failed regeneration cannot leave an old successful
            # trajectory (or a half-written replacement) behind.
            path.unlink(missing_ok=True)
            temporary_path.unlink(missing_ok=True)
            diagnostics = {"exception": type(exc).__name__, "message": str(exc)}
            _write_rejection(out_dir, name, case, diagnostics)
            record = _manifest_record(name, case, "exception")
            record["diagnostics"] = diagnostics
            records.append(record)
            print(f"[{i:03d}] REJECT {type(exc).__name__}: {exc}", flush=True)
        finally:
            # Each case has static JAX arguments.  Clear compiled executables so
            # a long sweep does not grow resident memory case by case.
            jax.clear_caches()

    manifest = _write_manifest(out_dir, args.set, records)
    print(
        f"manifest expected={manifest['expected']} accepted={manifest['accepted']} "
        f"rejected={manifest['rejected']} complete={str(manifest['complete']).lower()}",
        flush=True,
    )
    if args.require_complete and not manifest["complete"]:
        raise SystemExit("dataset generation failed --require-complete: manifest is incomplete")


if __name__ == "__main__":
    main()
