"""
Evaluate the surrogate's *transfer* performance.

Trained on simple cases (flat wall + periodic pillar arrays; optionally plus the
geometry-augmentation set), the surrogate is compared to the ground-truth
phase-field solver on BOTH the base simple train cases and the unseen complex test
surfaces of ``cases.CASES``.

Metrics:

  * **one-step** : given the *true* state at t, predict t+1 (no compounding error).
  * **rollout-K**: autoregressively predict K frames from the true initial frame:
    phase RMSE, liquid-mass error (relative to the initial mass), spreading-width
    error (length units) and the interface IoU of {phi > 0.5} at frame K.

Works with both legacy U-Net checkpoints (``train_surrogate.py``) and
``train_operator.py`` checkpoints (FNO/U-Net, chi/SDF geometry, conservative head).
All cases of a split are rolled out as one batch.  ``--ckpt persistence`` evaluates
the trivial "nothing moves" baseline, which is the reference for the one-step RMSE.

Usage::

    python evaluate_transfer.py --data data/base --ckpt ckpts/fno_sdf_large.pkl --horizon 40 \\
        --json results/fno_sdf_large.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import surrogate as S


def spreading(phi, L=6.0):
    m = phi > 0.5
    if not m.any():
        return 0.0
    xs = np.where(m.any(axis=1))[0]
    return (xs.max() - xs.min()) * (L / phi.shape[0])


def iou(a, b):
    a, b = a > 0.5, b > 0.5
    return float((a & b).sum() / max((a | b).sum(), 1))


def make_predictor(ckpt):
    if ckpt == "persistence":  # trivial baseline: phi(t+1) = phi(t)
        return (lambda state, geom, scal: state.copy()), dict(arch="persistence", geom="chi", uv_scale=1.0), "chi"
    model, params, cfg = S.load_checkpoint(ckpt)
    uv, geom_mode = cfg["uv_scale"], cfg.get("geom", "chi")
    apply_fn = jax.jit(lambda xb, cb: model.apply({"params": params}, xb, cb))

    def predict(state, geom, scal):
        """state (B,H,W,3) physical units -> next state (B,H,W,3)."""
        x = state.copy()
        x[..., 1:3] /= uv
        out = np.asarray(apply_fn(jnp.asarray(np.concatenate([x, geom], -1)), jnp.asarray(scal)))
        out = out.copy()
        out[..., 0] = np.clip(out[..., 0], 0.0, 1.0)
        out[..., 1:3] *= uv
        return out

    return predict, cfg, geom_mode


def evaluate(ckpt, data, horizon, verbose=True):
    predict, cfg, geom_mode = make_predictor(ckpt)
    rows, groups = [], defaultdict(list)
    for split in ("train", "test"):
        cs = S.load_full(data, split)
        if not cs:
            continue
        geoms, scals, trues = [], [], []
        for c in cs:
            geoms.append(S.geometry_features(c["chi"], c["sdf"], float(c["scalars"][3]), geom_mode))
            scals.append(c["scalars"])
            trues.append(np.stack([c["phi"], c["u"], c["v"]], -1))
        T = min(t.shape[0] for t in trues)
        true = np.stack([t[:T] for t in trues])  # (N,T,H,W,3)
        geom, scal = np.stack(geoms), np.stack(scals)
        N = true.shape[0]
        K = min(horizon, T)

        # one-step on true states (per case, all frames batched)
        one = []
        for i in range(N):
            pred = predict(true[i, :-1], np.repeat(geom[i : i + 1], T - 1, 0), np.repeat(scal[i : i + 1], T - 1, 0))
            one.append(float(np.sqrt(np.mean((pred[..., 0] - true[i, 1:, ..., 0]) ** 2))))

        # autoregressive rollout of all cases together (full length, metrics at K and T)
        st = true[:, 0].copy()
        traj = [st[..., 0]]
        for t in range(T - 1):
            st = predict(st, geom, scal)
            traj.append(st[..., 0])
        traj = np.stack(traj, 1)  # (N,T,H,W)

        for i, c in enumerate(cs):
            pt, tt = traj[i], true[i, ..., 0]
            fluid = (
                (c["sdf"] >= 0.0).astype(np.float32) if c["sdf"] is not None else (c["chi"] < 0.5).astype(np.float32)
            )
            m0 = max(float(np.sum(tt[0] * fluid)), 1e-6)
            eK = np.mean((pt[1:K] - tt[1:K]) ** 2)
            eT = np.mean((pt[1:] - tt[1:]) ** 2)
            mE = np.mean([abs(np.sum(pt[t] * fluid) - np.sum(tt[t] * fluid)) / m0 for t in range(1, K)])
            sE = np.mean([abs(spreading(pt[t]) - spreading(tt[t])) for t in range(1, K)])
            fam = "simple" if c["surface"] in ("flat", "pillars") else "complex"
            r = dict(
                split=split,
                family=fam,
                surface=c["surface"],
                file=os.path.basename(c["file"]),
                one_step=one[i],
                rollK=float(np.sqrt(eK)),
                rollT=float(np.sqrt(eT)),
                massK=float(mE),
                spreadK=float(sE),
                iouK=iou(pt[K - 1], tt[K - 1]),
                iouT=iou(pt[-1], tt[-1]),
            )
            rows.append(r)
            groups[(split, fam)].append(r)
            if verbose:
                print(
                    f"{split:5s} {c['surface']:15s} 1step={r['one_step']:.4f} roll{K}={r['rollK']:.4f} "
                    f"roll{T}={r['rollT']:.4f} mass={r['massK']:.3f} spread={r['spreadK']:.2f} IoU{K}={r['iouK']:.3f}",
                    flush=True,
                )
    summary = {}
    for key, rs in sorted(groups.items()):
        summary[f"{key[0]}/{key[1]}"] = {
            "n": len(rs),
            **{
                m: float(np.mean([r[m] for r in rs]))
                for m in ("one_step", "rollK", "rollT", "massK", "spreadK", "iouK", "iouT")
            },
        }
    return dict(
        ckpt=ckpt, cfg={k: v for k, v in cfg.items() if k != "train_hist"}, horizon=K, T=T, rows=rows, summary=summary
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/base")
    ap.add_argument("--ckpt", default="ckpts/surrogate.pkl")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    res = evaluate(args.ckpt, args.data, args.horizon)
    K = res["horizon"]
    print(f"\n-- grouped: {args.ckpt} --")
    print(
        f"{'group':16s} {'n':>3s} {'1step':>7s} {'roll' + str(K):>8s} "
        f"{'roll' + str(res['T']):>8s} {'mass':>6s} {'spread':>7s} {'IoU' + str(K):>7s}"
    )
    for g, s in res["summary"].items():
        print(
            f"{g:16s} {s['n']:3d} {s['one_step']:7.4f} {s['rollK']:8.4f} {s['rollT']:8.4f} "
            f"{s['massK']:6.3f} {s['spreadK']:7.2f} {s['iouK']:7.3f}"
        )
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
