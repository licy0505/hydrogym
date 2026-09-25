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
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import cases as C
import phasefield as pf


def _case_name(case: dict, set_name: str, index: int) -> str:
    """Return a stable, filesystem-safe case name."""
    if set_name == "base":
        label = C.case_label({k: v for k, v in case.items() if k != "family"})
        return f"{case['split']}_{index:02d}_{label}"
    return f"{case['split']}_{set_name}_{index:03d}_{case['surface']}"


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


def _diagnose(
    initial: pf.State,
    phi: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    solid: pf.Solid,
    *,
    max_phi_overshoot: float,
    max_solid_leak: float,
    min_total_mass_ratio: float,
    max_total_mass_ratio: float,
    max_speed: float,
) -> tuple[bool, dict]:
    """Validate finite values, mass, phase bounds, and solid leakage."""
    chi = np.asarray(solid.chi, dtype=np.float64)
    phi0 = np.asarray(initial.phi, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    total0 = float(np.mean(phi0))
    total = np.mean(phi, axis=(1, 2))
    denom = np.maximum(np.mean(np.abs(phi), axis=(1, 2)), 1e-12)
    leak = np.mean(np.abs(phi) * chi[None], axis=(1, 2)) / denom
    speed = np.sqrt(u * u + v * v)

    finite = bool(np.isfinite(phi).all() and np.isfinite(u).all() and np.isfinite(v).all())
    overshoot = float(max(np.max(-phi), np.max(phi - 1.0), 0.0)) if phi.size else 0.0
    total_ratio = total / max(total0, 1e-12)
    diag = {
        "finite": finite,
        "initial_total_mass": total0,
        "final_total_mass_ratio": float(total_ratio[-1]),
        "min_total_mass_ratio": float(np.min(total_ratio)),
        "max_total_mass_ratio": float(np.max(total_ratio)),
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
        and diag["max_solid_leak"] <= max_solid_leak
        and diag["max_phi_overshoot"] <= max_phi_overshoot
        and diag["max_speed"] <= max_speed
    )
    return bool(ok), diag


def _write_rejection(out_dir: Path, name: str, case: dict, diagnostics: dict) -> None:
    reject_dir = out_dir / "rejected"
    reject_dir.mkdir(parents=True, exist_ok=True)
    payload = {"case": case, "diagnostics": diagnostics}
    (reject_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


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
) -> None:
    """Write one validated trajectory with explicit physical-time metadata."""
    phi_d = _downsample_history(phi, ds).astype(np.float16)
    u_d = _downsample_history(u, ds).astype(np.float16)
    v_d = _downsample_history(v, ds).astype(np.float16)
    chi_d = _downsample_history(np.asarray(solid.chi)[None], ds)[0].astype(np.float16)
    sdf = np.asarray(jnp.clip(solid.sdf, -1.0, 3.0))[None]
    sdf_d = _downsample_history(sdf, ds)[0].astype(np.float32)
    scalars = np.array(
        [case.get("We", 100.0) / 100.0, case.get("Re", 200.0) / 200.0, float(case.get("cos_theta", 0.0)), p.dx],
        dtype=np.float32,
    )
    case_meta = dict(case)
    case_meta.update(
        solver_dt=float(p.dt),
        save_every=int(save_every),
        frame_dt=float(p.dt * save_every),
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
    ap.add_argument("--dry-run", action="store_true", help="build cases and print schedule without integrating")
    ap.add_argument("--max-phi-overshoot", type=float, default=0.08)
    ap.add_argument("--max-solid-leak", type=float, default=0.08)
    ap.add_argument("--min-total-mass-ratio", type=float, default=0.95)
    ap.add_argument("--max-total-mass-ratio", type=float, default=1.05)
    ap.add_argument("--max-speed", type=float, default=5.0)
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

    for i, case in enumerate(all_cases):
        name = _case_name(case, args.set, i)
        path = out_dir / f"{name}.npz"
        if path.exists() and not args.overwrite:
            print(f"[{i:03d}] exists, skip: {path.name}", flush=True)
            continue

        dt, nsteps, save_every = _effective_schedule(case, args)
        print(f"[{i:03d}] {name}: dt={dt:g}, nsteps={nsteps}, save_every={save_every}", flush=True)
        if args.dry_run:
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
                max_phi_overshoot=args.max_phi_overshoot,
                max_solid_leak=args.max_solid_leak,
                min_total_mass_ratio=args.min_total_mass_ratio,
                max_total_mass_ratio=args.max_total_mass_ratio,
                max_speed=args.max_speed,
            )
            if not ok:
                _write_rejection(out_dir, name, case, diagnostics)
                print(f"[{i:03d}] REJECT {diagnostics} -> rejected/{name}.json", flush=True)
                continue
            _save_case(path, case, p, solid, np.asarray(phi), np.asarray(u), np.asarray(v), save_every, args.ds, diagnostics)
            print(f"[{i:03d}] saved T={phi.shape[0]} in {time.time() - t0:.1f}s -> {path}", flush=True)
        except (FloatingPointError, ValueError, RuntimeError) as exc:
            diagnostics = {"exception": type(exc).__name__, "message": str(exc)}
            _write_rejection(out_dir, name, case, diagnostics)
            print(f"[{i:03d}] REJECT {type(exc).__name__}: {exc}", flush=True)
        finally:
            # Each case has static JAX arguments.  Clear compiled executables so
            # a long sweep does not grow resident memory case by case.
            jax.clear_caches()


if __name__ == "__main__":
    main()
