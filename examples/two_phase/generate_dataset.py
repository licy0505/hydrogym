"""
Generate a droplet-impact dataset for the phase-distribution surrogate.

Runs the phase-field solver over the train/test cases in ``cases.py``, saving
downsampled (phi, u, v) trajectories plus the solid indicator and scalar
parameters per case, as ``.npz`` files under ``data/`` (gitignored).

Usage::

    python generate_dataset.py --out data --nsteps 2000 --ds 3

The ``--ds`` factor controls downsampling (192/ds -> network resolution).
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

import cases as C
import phasefield as pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--nsteps", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--ds", type=int, default=3)
    ap.add_argument("--N", type=int, default=192, help="solver resolution")
    ap.add_argument("--dt", type=float, default=4e-3, help="solver timestep")
    ap.add_argument("--limit", type=int, default=0, help="max number of cases (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    allcases = C.CASES
    if args.limit:
        allcases = allcases[: args.limit]

    for i, case in enumerate(allcases):
        label = C.case_label(case)
        path = os.path.join(args.out, f"{case['split']}_{i:02d}_{label}.npz")
        if os.path.exists(path):
            print(f"[{i:2d}] exists, skip: {label}")
            continue
        p, solid, st = pf.build_case(case, N=args.N, dt=args.dt)
        final, phi, u, v = pf.rollout(st, solid, p, args.nsteps, save_every=args.save_every)
        f = args.ds
        assert p.Nx % f == 0, "ds must divide N"
        phi_d = np.stack([np.asarray(pf.downsample(phi[t], f)) for t in range(phi.shape[0])])
        u_d = np.stack([np.asarray(pf.downsample(u[t], f)) for t in range(u.shape[0])])
        v_d = np.stack([np.asarray(pf.downsample(v[t], f)) for t in range(v.shape[0])])
        chi_d = np.asarray(pf.downsample(solid.chi, f))
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
            scalars=scalars,
            surface=np.array(case.get("surface", "flat")),
            split=np.array(case["split"]),
        )
        print(f"[{i:2d}] {case['split']:4s} {label}  T={phi_d.shape[0]} -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
