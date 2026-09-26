"""
Train the FNO / U-Net surrogate (with optional SDF geometry encoding) on large datasets.

Compared with ``train_surrogate.py`` (legacy U-Net, whole dataset in memory) this
script

* streams windows from a float16 :class:`surrogate.TrajectoryStore`, so it scales
  to hundreds of trajectories on a laptop-sized machine,
* is step-based (``--steps``) with warm-up + cosine LR, AdamW and grad clipping,
* supports ``--arch fno|unet``, ``--geom sdf|chi``, residual prediction and the
  exact mass-conservative head,
* supports unrolled training (``--unroll K``) and resuming (``--resume``), so the
  usual recipe is teacher-forced pre-training followed by an unrolled fine-tune.

Examples::

    # FNO + SDF on the large simple set (+ the 13 base simple cases)
    python train_operator.py --data data/base,data/large --arch fno --geom sdf \\
        --steps 3000 --out ckpts/fno_sdf_large.pkl
    # unrolled fine-tune
    python train_operator.py --data data/base,data/large --resume ckpts/fno_sdf_large.pkl \\
        --unroll 3 --steps 400 --batch 8 --lr 3e-4 --out ckpts/fno_sdf_large_u3.pkl
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--data", default="data/base,data/large", help="comma-separated dataset dirs")
    ap.add_argument("--families", default="", help="restrict train families, e.g. 'simple'")
    ap.add_argument("--arch", default="fno", choices=["fno", "unet"])
    ap.add_argument("--geom", default="sdf", choices=["sdf", "chi"])
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--no-conservative", action="store_true")
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--unroll", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.0, help="input-noise std (phi units) for robustness")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out", default="ckpts/fno_sdf.pkl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    params = None
    if args.resume:
        _, params, cfg = S.load_checkpoint(args.resume)
        assert not cfg.get("legacy"), "resume a train_operator.py checkpoint"
        assert cfg.get("dataset_schema_version") == S.DATASET_SCHEMA_VERSION
        print(f"resumed {args.resume}: {cfg}", flush=True)
    else:
        cfg = dict(
            dataset_schema_version=S.DATASET_SCHEMA_VERSION,
            arch=args.arch,
            geom=args.geom,
            residual=not args.no_residual,
            conservative=not args.no_conservative,
            width=args.width,
            modes=args.modes,
            layers=args.layers,
            base=args.base,
            levels=args.levels,
        )
    fams = tuple(f for f in args.families.split(",") if f) or None
    store = S.TrajectoryStore(args.data, "train", geom=cfg["geom"], families=fams)
    uv = cfg.get("uv_scale") or store.uv_scale
    cfg["uv_scale"] = uv
    cfg["data"] = args.data
    cfg["n_train_traj"] = len(store)
    K = args.unroll
    win = store.windows(K)
    fam_counts = {f: sum(m["family"] == f for m in store.meta) for f in ("simple", "complex")}
    print(f"trajectories={len(store)} {fam_counts}  T={store.T}  windows(K={K})={len(win)}  uv={uv:.3f}", flush=True)

    model = S.build_model(cfg)
    G = store.geom.shape[-1]
    H, W = store.states.shape[2:4]
    if params is None:
        params = model.init(jax.random.PRNGKey(args.seed), jnp.ones((1, H, W, 3 + G)), jnp.ones((1, 4)))["params"]
    npar = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"model={cfg['arch']} geom={cfg['geom']} params={npar}", flush=True)

    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, min(200, args.steps // 10), args.steps, args.lr * 0.02)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=args.wd))
    opt_state = tx.init(params)
    wts = jnp.array([2.0, 1.0, 1.0])

    def loss_fn(p, fr, geom, cond, key):
        state = fr[:, 0]
        if args.noise > 0:
            state = state + args.noise * jax.random.normal(key, state.shape)
        tot = 0.0
        for k in range(1, fr.shape[1]):
            pred = model.apply({"params": p}, jnp.concatenate([state, geom], axis=-1), cond)
            tot = tot + jnp.mean(wts * (pred - fr[:, k]) ** 2)
            state = pred.at[..., 0].set(jnp.clip(pred[..., 0], 0.0, 1.0))
        return tot / (fr.shape[1] - 1)

    @jax.jit
    def train_step(p, o, fr, geom, cond, key):
        lv, g = jax.value_and_grad(loss_fn)(p, fr, geom, cond, key)
        upd, o = tx.update(g, o, p)
        return optax.apply_updates(p, upd), o, lv

    def save():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "wb") as fh:
            pickle.dump(dict(params=params, cfg=cfg, uv_scale=uv), fh)

    t0, run, hist = time.time(), 0.0, []
    key = jax.random.PRNGKey(args.seed + 1)
    for it in range(1, args.steps + 1):
        b = win[rng.integers(0, len(win), size=args.batch)]
        fr, geom, cond = store.batch(b, K, uv)
        key, sk = jax.random.split(key)
        params, opt_state, lv = train_step(params, opt_state, jnp.asarray(fr), jnp.asarray(geom), jnp.asarray(cond), sk)
        run += float(lv)
        if it % args.log_every == 0:
            hist.append((it, run / args.log_every))
            el = time.time() - t0
            print(
                f"step {it:5d}  loss={run / args.log_every:.3e}  ({el:.0f}s, eta {el / it * (args.steps - it):.0f}s)",
                flush=True,
            )
            run = 0.0
        if it % 500 == 0:
            save()
    cfg["train_hist"] = hist
    save()
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump({k: v for k, v in cfg.items()}, fh, indent=1)
    print(f"saved {args.out}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
