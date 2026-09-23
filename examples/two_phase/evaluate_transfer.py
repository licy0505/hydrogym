"""
Evaluate the surrogate's *transfer* performance.

Trained only on simple cases (flat wall + a couple of pillar arrays), the
surrogate is compared to the ground-truth phase-field solver on BOTH train and
unseen complex test surfaces.

Two complementary metrics:

  * **one-step** : given the *true* state at t, predict t+1.  This measures how
    well the learned operator generalises to unseen geometry for predicting the
    next phase distribution (no compounding error).
  * **rollout-K**: autoregressively predict the first K frames from the true
    initial frame.  This shows how error accumulates over the impact event.

Reported per surface family: phase RMSE, and (rollout) liquid-mass / spreading
relative errors.  The train-vs-test gap is the quantitative answer to
"does it generalise from simple cases to complex surfaces?"

Usage::

    python evaluate_transfer.py --data data --ckpt ckpts/surrogate.pkl --horizon 40
"""

from __future__ import annotations

import argparse
import pickle
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


def liquid(phi):
    return float(phi.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="ckpts/surrogate.pkl")
    ap.add_argument("--horizon", type=int, default=40)
    args = ap.parse_args()

    with open(args.ckpt, "rb") as fh:
        ck = pickle.load(fh)
    params, uv = ck["params"], ck["uv_scale"]
    model = S.UNet(base=16, levels=3, out_channels=3)
    apply_fn = jax.jit(lambda xb, cb: model.apply({"params": params}, xb, cb))

    def predict(phi, u, v, chi, scal):
        x = np.stack([phi, u / uv, v / uv, chi], axis=-1)[None]
        out = np.asarray(apply_fn(jnp.asarray(x), jnp.asarray(scal[None])))[0]
        return (np.clip(out[..., 0], 0.0, 1.0), out[..., 1] * uv, out[..., 2] * uv)

    fam1 = defaultdict(list)  # one-step phi rmse
    famK = defaultdict(list)  # rollout-K metrics
    print(
        f"{'split':5s} {'surface':15s} {'1step_RMSE':>10s} {'rollK_RMSE':>10s} "
        f"{'rollK_mass':>10s} {'rollK_spread':>12s}"
    )
    for split in ("train", "test"):
        for c in S.load_full(args.data, split):
            phi_t, u_t, v_t, chi, scal = (c["phi"], c["u"], c["v"], c["chi"], c["scalars"])
            T = phi_t.shape[0]
            # --- one-step on true states ---
            err1 = []
            for t in range(T - 1):
                pp, _, _ = predict(phi_t[t], u_t[t], v_t[t], chi, scal)
                err1.append(np.mean((pp - phi_t[t + 1]) ** 2))
            one = float(np.sqrt(np.mean(err1)))
            # --- autoregressive rollout, first K frames ---
            K = min(args.horizon, T)
            m0 = max(liquid(phi_t[0]), 1e-6)
            phi_p, u_p, v_p = phi_t[0].copy(), u_t[0].copy(), v_t[0].copy()
            eK, mE, sE = [], [], []
            for t in range(K - 1):
                phi_p, u_p, v_p = predict(phi_p, u_p, v_p, chi, scal)
                eK.append(np.mean((phi_p - phi_t[t + 1]) ** 2))
                mE.append(abs(liquid(phi_p) - liquid(phi_t[t + 1])) / m0)  # rel. to initial mass
                sE.append(abs(spreading(phi_p) - spreading(phi_t[t + 1])))  # length units
            rk = float(np.sqrt(np.mean(eK)))
            rm = float(np.mean(mE))
            rs = float(np.mean(sE))
            fam1[(split, "simple" if c["surface"] in ("flat", "pillars") else "complex")].append(one)
            famK[(split, "simple" if c["surface"] in ("flat", "pillars") else "complex")].append((rk, rm, rs))
            print(f"{split:5s} {c['surface']:15s} {one:10.4f} {rk:10.4f} {rm:10.1f}% {rs:12.1f}%")

    print("\n-- grouped (one-step RMSE / rollout-K) --")
    for key in sorted(set(fam1) | set(famK)):
        o = np.mean(fam1.get(key, [np.nan]))
        v = np.array(famK.get(key, [(np.nan, np.nan, np.nan)]))
        print(
            f"{key[0]:5s} {key[1]:8s}  n={len(fam1.get(key, [])):2d}  1step={o:.4f}  "
            f"rollK_RMSE={v[:, 0].mean():.4f} mass_rel={v[:, 1].mean():6.2f} spread_abs={v[:, 2].mean():6.2f}"
        )


if __name__ == "__main__":
    main()
