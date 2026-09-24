"""
Generate a droplet-impact dataset for the phase-distribution surrogate.

Runs the phase-field solver over a case set from ``cases.py``, saving
downsampled (phi, u, v) trajectories plus the solid indicator, the solid
signed-distance field (SDF) and scalar parameters per case, as ``.npz`` files
(gitignored).

Case sets (``--set``):

* ``base``  : the 13 simple train + 10 complex test cases of ``cases.CASES``
* ``large`` : 120 procedurally sampled *simple* cases (flat + periodic pillars)
* ``aug``   : 48 geometry-augmentation cases (complex families, seeds disjoint
  from the test set)

Usage::

    python generate_dataset.py --set base  --out data/base  --nsteps 2000 --ds 3
    python generate_dataset.py --set large --out data/large --nsteps 2000 --ds 3
    python generate_dataset.py --set aug   --out data/aug   --nsteps 2000 --ds 3

The ``--ds`` factor controls downsampling (192/ds -> network resolution).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import cases as C
import phasefield as pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="base", choices=sorted(C.CASE_SETS))
    ap.add_argument("--out", default="data")
    ap.add_argument("--nsteps", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--ds", type=int, default=3)
    ap.add_argument("--N", type=int, default=192, help="solver resolution")
    ap.add_argument("--dt", type=float, default=4e-3, help="solver timestep")
    ap.add_argument("--limit", type=int, default=0, help="max number of cases (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    allcases = C.CASE_SETS[args.set]()
    if args.limit:
        allcases = allcases[: args.limit]

    for i, case in enumerate(allcases):
        label = C.case_label({k: v for k, v in case.items() if k != "family"})
        name = f"{case['split']}_{i:02d}_{label}" if args.set == "base" else f"{case['split']}_{args.set}_{i:03d}_{case['surface']}"
        path = os.path.join(args.out, name + ".npz")
        if os.path.exists(path):
            print(f"[{i:2d}] exists, skip: {label}")
            continue
        t0 = time.time()
        p, solid, st = pf.build_case(case, N=args.N, dt=args.dt)
        final, phi, u, v = pf.rollout(st, solid, p, args.nsteps, save_every=args.save_every)
        if not bool(np.isfinite(np.asarray(phi)).all()):
            print(f"[{i:2d}] NaN/inf in trajectory, skip: {label}", flush=True)
            continue
        f = args.ds
        assert p.Nx % f == 0, "ds must divide N"
        phi_d = np.stack([np.asarray(pf.downsample(phi[t], f)) for t in range(phi.shape[0])])
        u_d = np.stack([np.asarray(pf.downsample(u[t], f)) for t in range(u.shape[0])])
        v_d = np.stack([np.asarray(pf.downsample(v[t], f)) for t in range(v.shape[0])])
        chi_d = np.asarray(pf.downsample(solid.chi, f))
        # SDF: clip far field (irrelevant + keeps float16 precise), then pool
        sdf_d = np.asarray(pf.downsample(jnp.clip(solid.sdf, -1.0, 3.0), f))
        scalars = np.array(
            [case.get("We", 100.0) / 100.0, case.get("Re", 200.0) / 200.0, float(case.get("cos_theta", 0.0)), p.dx],
            dtype=np.float32,
        )
        np.savez_compressed(
            path,
            phi=phi_d.astype(np.float16),
            u=u_d.astype(np.float16),
            v=v_d.astype(np.float16),
            chi=chi_d.astype(np.float16),
            sdf=sdf_d.astype(np.float32),
            scalars=scalars,
            surface=np.array(case.get("surface", "flat")),
            split=np.array(case["split"]),
            case=np.array(json.dumps(case)),
        )
        print(
            f"[{i:2d}] {case['split']:4s} {label}  T={phi_d.shape[0]} ({time.time() - t0:.1f}s) -> {os.path.basename(path)}",
            flush=True,
        )
        # every case has its own (static) params -> its own compiled rollout; drop the
        # executables so memory stays flat over hundreds of cases
        del final, phi, u, v
        jax.clear_caches()


if __name__ == "__main__":
    main()
