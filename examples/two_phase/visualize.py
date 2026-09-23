"""
Generate publication-style visualizations for the two-phase droplet-impact example.

Two modes:

* ``--mode solver``   (no model needed): showcases the *solver* --
    fig_surfaces.png : droplet impact across six surface families
    fig_weber.png    : Weber-number sweep on a flat wall
    fig_wetting.png  : wettability (cos theta) sweep on a flat wall

* ``--mode transfer`` (needs ckpts/surrogate.pkl): showcases the *surrogate* --
    fig_transfer.png : autoregressive rollout (blue) vs solver truth (red)
    fig_metrics.png  : spreading D(t) / liquid-mass curves + one-step RMSE bars

All figures are written to ``figures/`` (committed to the repo as documentation).

Usage::

    python visualize.py --mode solver
    python visualize.py --mode transfer --ckpt ckpts/surrogate.pkl
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cases as C
import phasefield as pf

OSDIR = "figures"


def flip(a):
    """Orient arrays with the wall at the bottom for display."""
    return np.flipud(np.asarray(a).T)


def overlay(ax, phi, chi, title=None, truth=None):
    ax.imshow(flip(chi), cmap="gray_r", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(flip(phi) < 0.5, flip(phi)), cmap="Blues", vmin=0, vmax=1, alpha=0.9)
    if truth is not None:
        ax.contour(flip(truth), levels=[0.5], colors="r", linewidths=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def run_case(case, nsteps=1200, save_every=10):
    p, solid, st = pf.build_case(case)
    _, phi, u, v = pf.rollout(st, solid, p, nsteps, save_every=save_every)
    return p, solid, np.asarray(phi), np.asarray(u), np.asarray(v)


# -------------------------------------------------------------------------------------
#  solver figures
# -------------------------------------------------------------------------------------


def fig_surfaces():
    surfaces = ["flat", "pillars", "random_pillars", "hierarchical", "grooves", "wedge"]
    frames = [5, 25, 45, 70, 100]
    fig, axes = plt.subplots(len(surfaces), len(frames), figsize=(2.2 * len(frames), 2.3 * len(surfaces)))
    for r, s in enumerate(surfaces):
        case = dict(surface=s, We=150.0, cos_theta=0.0, seed=7, n_pillars=5)
        p, solid, phi, _, _ = run_case(case)
        T = phi.shape[0]
        for c, ti in enumerate(frames):
            ti = min(ti, T - 1)
            overlay(axes[r][c], phi[ti], solid.chi, title=f"t={ti}" if r == 0 else None)
            if c == 0:
                axes[r][c].set_ylabel(s, fontsize=9)
    fig.suptitle("Droplet impact across surface families (We=150)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OSDIR}/fig_surfaces.png", dpi=110)
    plt.close(fig)
    print("saved fig_surfaces.png")


def fig_weber():
    Wes = [50.0, 100.0, 200.0, 300.0]
    frames = [10, 30, 60, 100]
    fig, axes = plt.subplots(len(Wes), len(frames), figsize=(2.2 * len(frames), 2.3 * len(Wes)))
    for r, we in enumerate(Wes):
        case = dict(surface="flat", We=we, cos_theta=0.0)
        p, solid, phi, _, _ = run_case(case)
        T = phi.shape[0]
        for c, ti in enumerate(frames):
            ti = min(ti, T - 1)
            overlay(axes[r][c], phi[ti], solid.chi, title=f"t={ti}" if r == 0 else None)
            if c == 0:
                axes[r][c].set_ylabel(f"We={we:g}", fontsize=9)
    fig.suptitle("Weber-number sweep on a flat wall", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OSDIR}/fig_weber.png", dpi=110)
    plt.close(fig)
    print("saved fig_weber.png")


def fig_wetting():
    cts = [-0.99, 0.0, 0.99]
    frames = [10, 40, 80, 110]
    fig, axes = plt.subplots(len(cts), len(frames), figsize=(2.2 * len(frames), 2.3 * len(cts)))
    for r, ct in enumerate(cts):
        case = dict(surface="flat", We=100.0, cos_theta=ct)
        p, solid, phi, _, _ = run_case(case)
        T = phi.shape[0]
        for c, ti in enumerate(frames):
            ti = min(ti, T - 1)
            overlay(axes[r][c], phi[ti], solid.chi, title=f"t={ti}" if r == 0 else None)
            if c == 0:
                axes[r][c].set_ylabel(f"cos$\\theta$={ct}", fontsize=9)
    fig.suptitle("Wettability sweep: hydrophobic -> hydrophilic (We=100)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OSDIR}/fig_wetting.png", dpi=110)
    plt.close(fig)
    print("saved fig_wetting.png")


# -------------------------------------------------------------------------------------
#  transfer figures
# -------------------------------------------------------------------------------------


def load_model(ckpt):
    import pickle

    import surrogate as S

    with open(ckpt, "rb") as fh:
        ck = pickle.load(fh)
    params, uv = ck["params"], ck["uv_scale"]
    base = ck.get("base", 16)
    levels = ck.get("levels", 3)
    model = S.UNet(base=base, levels=levels, out_channels=3)
    apply_fn = jax.jit(lambda xb, cb: model.apply({"params": params}, xb, cb))
    return apply_fn, uv


def rollout_surrogate(apply_fn, uv, c, K):
    phi_t, u_t, v_t, chi, scal = c["phi"], c["u"], c["v"], c["chi"], c["scalars"]
    phi_p, u_p, v_p = phi_t[0].copy(), u_t[0].copy(), v_t[0].copy()
    out = [phi_p]
    for t in range(K - 1):
        x = np.stack([phi_p, u_p / uv, v_p / uv, chi], -1)[None]
        o = np.asarray(apply_fn(jnp.asarray(x), jnp.asarray(scal[None])))[0]
        phi_p = np.clip(o[..., 0], 0, 1)
        u_p, v_p = o[..., 1] * uv, o[..., 2] * uv
        out.append(phi_p.copy())
    return np.stack(out)


def fig_transfer(ckpt):
    import surrogate as S

    apply_fn, uv = load_model(ckpt)
    picks = (
        [c for c in S.load_full("data", "train") if c["surface"] == "flat"][:1]
        + [c for c in S.load_full("data", "test") if c["surface"] == "random_pillars"][:1]
        + [c for c in S.load_full("data", "test") if c["surface"] == "hierarchical"][:1]
    )
    frames = [0, 10, 20, 30, 39]
    fig, axes = plt.subplots(len(picks), len(frames), figsize=(2.2 * len(frames), 2.4 * len(picks)))
    for r, c in enumerate(picks):
        pred = rollout_surrogate(apply_fn, uv, c, 40)
        for ci, ti in enumerate(frames):
            overlay(axes[r][ci], pred[ti], c["chi"], title=f"t={ti}" if r == 0 else None, truth=c["phi"][ti])
            if ci == 0:
                axes[r][ci].set_ylabel(c["surface"], fontsize=9)
    fig.suptitle("Surrogate rollout (blue) vs solver truth (red) -- trained on simple only", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OSDIR}/fig_transfer.png", dpi=110)
    plt.close(fig)
    print("saved fig_transfer.png")


def spreading(phi, L=6.0):
    m = phi > 0.5
    if not m.any():
        return 0.0
    xs = np.where(m.any(axis=1))[0]
    return (xs.max() - xs.min()) * (L / phi.shape[0])


def fig_metrics(ckpt):
    import surrogate as S

    apply_fn, uv = load_model(ckpt)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    tests = [c for c in S.load_full("data", "test")]
    for i, c in enumerate(tests[:3]):
        K = 40
        pred = rollout_surrogate(apply_fn, uv, c, K)
        tt = np.arange(K)
        d_true = [spreading(c["phi"][t]) for t in range(K)]
        d_pred = [spreading(pred[t]) for t in range(K)]
        m_true = [c["phi"][t].sum() for t in range(K)]
        m_pred = [pred[t].sum() for t in range(K)]
        axes[0][i].plot(tt, d_true, "r-", label="truth")
        axes[0][i].plot(tt, d_pred, "b--", label="surrogate")
        axes[0][i].set_title(f"D(t) {c['surface']}", fontsize=9)
        axes[1][i].plot(tt, m_true, "r-")
        axes[1][i].plot(tt, m_pred, "b--")
        axes[1][i].set_title(f"mass {c['surface']}", fontsize=9)
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Transfer: spreading width and liquid mass, surrogate vs truth (3 unseen surfaces)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OSDIR}/fig_metrics.png", dpi=110)
    plt.close(fig)
    print("saved fig_metrics.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["solver", "transfer"], default="solver")
    ap.add_argument("--ckpt", default="ckpts/surrogate.pkl")
    args = ap.parse_args()
    os.makedirs(OSDIR, exist_ok=True)
    if args.mode == "solver":
        fig_surfaces()
        fig_weber()
        fig_wetting()
    else:
        fig_transfer(args.ckpt)
        fig_metrics(args.ckpt)


if __name__ == "__main__":
    main()
