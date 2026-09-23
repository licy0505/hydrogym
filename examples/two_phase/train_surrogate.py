"""
Train the conditional U-Net surrogate on the *simple* (train) cases.

Two training objectives:

* ``--unroll 1`` (default): teacher-forced one-step MSE.
* ``--unroll K>1``          : *unrolled* training -- the model is rolled out K steps
  on its own predictions and the MSE is accumulated over the trajectory, so
  gradients flow through the autoregressive loop.  This reduces exposure bias and
  noticeably improves multi-step rollout accuracy (see README / evaluation).

Optionally resume from an existing checkpoint with ``--resume`` (e.g. fine-tune a
teacher-forced model with a few unrolled epochs).

Usage::

    python train_surrogate.py --epochs 25                       # teacher-forced
    python train_surrogate.py --epochs 6 --unroll 3 --lr 3e-4 \\
        --resume ckpts/surrogate.pkl --out ckpts/surrogate_unrolled.pkl

Writes parameters to ``--out`` (default ckpts/surrogate.pkl).
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import surrogate as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--unroll", type=int, default=1)
    ap.add_argument("--mass-lam", type=float, default=0.0, help="mass-conservation penalty weight")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default="ckpts/surrogate.pkl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    uv = None
    params = None

    model = S.UNet(base=args.base, levels=args.levels, out_channels=3)

    if args.resume and os.path.exists(args.resume):
        with open(args.resume, "rb") as fh:
            ck = pickle.load(fh)
        params = ck["params"]
        uv = ck["uv_scale"]
        print(f"resumed from {args.resume} (uv={uv:.3f})", flush=True)

    # ---------------------------------------------------------------- data
    if args.unroll > 1:
        wins = S.load_windows(args.data, "train", args.unroll)
        fr = np.stack([w["frames"] for w in wins])
        chi = np.stack([w["chi"] for w in wins])
        scal = np.stack([w["scalars"] for w in wins])
        if uv is None:
            uv = float(np.maximum(np.abs(fr[..., 1:3]).max(), 1e-3))
        fr = fr.copy()
        fr[..., 1:3] /= uv
        n = len(wins)
        print(f"unroll={args.unroll}  windows: {fr.shape}  uv_scale={uv:.3f}", flush=True)
    else:
        X, Y, C, _ = S.load_arrays(args.data, "train")
        if uv is None:
            uv = float(np.maximum(np.abs(X[..., 1:3]).max(), 1e-3))
        X[..., 1:3] /= uv
        Y[..., 1:3] /= uv
        n = X.shape[0]
        print(f"train samples: {X.shape}  uv_scale={uv:.3f}", flush=True)

    if params is None:
        params = model.init(jax.random.PRNGKey(args.seed), jnp.ones((1, 64, 64, 4)), jnp.ones((1, 4)))["params"]
    print(f"model params: {sum(x.size for x in jax.tree_util.tree_leaves(params))}", flush=True)

    tx = optax.adam(args.lr)
    opt_state = tx.init(params)
    w = jnp.array([2.0, 1.0, 1.0])
    ml = args.mass_lam

    def mass_term(a, b):
        # mean liquid fraction difference (mass-conservation penalty)
        return jnp.mean((a.mean(axis=(1, 2)) - b.mean(axis=(1, 2))) ** 2)

    if args.unroll > 1:
        fr_j, chi_j, scal_j = jnp.asarray(fr), jnp.asarray(chi), jnp.asarray(scal)

        def loss_fn(p, fb, chib, cb):
            total = 0.0
            phi, u, v = fb[:, 0, ..., 0], fb[:, 0, ..., 1], fb[:, 0, ..., 2]
            for k in range(1, fb.shape[1]):
                x = jnp.stack([phi, u, v, chib], axis=-1)
                pred = model.apply({"params": p}, x, cb)
                total += jnp.mean(w * (pred - fb[:, k]) ** 2)
                if ml > 0:
                    total += ml * mass_term(pred[..., 0], fb[:, k, ..., 0])
                phi = jnp.clip(pred[..., 0], 0.0, 1.0)
                u, v = pred[..., 1], pred[..., 2]
            return total / (fb.shape[1] - 1)

        gv = jax.jit(jax.value_and_grad(loss_fn))

        def step(p, o, fb, chib, cb):
            lv, g = gv(p, fb, chib, cb)
            u, o = tx.update(g, o, p)
            return optax.apply_updates(p, u), o, lv

        step = jax.jit(step)
        get = lambda b: (fr_j[b], chi_j[b], scal_j[b])
    else:
        X_j, Y_j, C_j = jnp.asarray(X), jnp.asarray(Y), jnp.asarray(C)

        def loss_fn(p, xb, yb, cb):
            pred = model.apply({"params": p}, xb, cb)
            loss = jnp.mean(w * (pred - yb) ** 2)
            if ml > 0:
                loss = loss + ml * mass_term(pred[..., 0], yb[..., 0])
            return loss

        gv = jax.jit(jax.value_and_grad(loss_fn))

        def step(p, o, xb, yb, cb):
            lv, g = gv(p, xb, yb, cb)
            u, o = tx.update(g, o, p)
            return optax.apply_updates(p, u), o, lv

        step = jax.jit(step)
        get = lambda b: (X_j[b], Y_j[b], C_j[b])

    t0 = time.time()
    for ep in range(args.epochs):
        idx = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, args.batch):
            b = idx[s : s + args.batch]
            if len(b) < 4:
                continue
            params, opt_state, lv = step(params, opt_state, *get(b))
            tot += float(lv) * len(b)
        print(f"epoch {ep:2d}  loss={tot / n:.5f}  ({time.time() - t0:.0f}s)", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "wb") as fh:
            pickle.dump(dict(params=params, uv_scale=float(uv), base=args.base, levels=args.levels), fh)

    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
