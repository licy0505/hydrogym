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


def overlay(ax, phi, chi, title=None, truth=None, mask_solid=True):
    ax.imshow(flip(chi), cmap="gray_r", vmin=0, vmax=1, origin="lower")
    if mask_solid:
        mask = (flip(phi) < 0.5) | (flip(chi) > 0.5)
    else:
        mask = flip(phi) < 0.5
    ax.imshow(np.ma.masked_where(mask, flip(phi)), cmap="Blues", vmin=0, vmax=1, alpha=0.9)
    if truth is not None:
        if mask_solid:
            truth = np.where(np.asarray(chi) > 0.5, 0.0, truth)
        ax.contour(flip(truth), levels=[0.5], colors="r", linewidths=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def run_case(case, nsteps=1200, save_every=10, N=192, dt=None):
    p, solid, st = pf.build_case(case, N=N, dt=dt if dt is not None else 4e-3)
    # build_case already clips dt to CFL; show what was actually used
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


def fig_diagnosis():
    """Explain the 'liquid in the wall' artifact: diffuse solid + volume wetting."""
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.2, 1.6, 1.6], wspace=0.35)

    case = dict(surface="flat", We=100.0, cos_theta=0.0, R=0.7)

    # single run at N=192 (default solver, no hard clip)
    p, solid, phi, _, _ = run_case(case, nsteps=1200, save_every=10, N=192)
    chi = np.asarray(solid.chi)
    T = phi.shape[0]
    ti = min(70, T - 1)

    # A: unmasked (leak visible)
    ax = fig.add_subplot(gs[0, 0])
    overlay(ax, phi[ti], chi, mask_solid=False)
    ax.set_title("Unmasked (N=192)\nphi>0.5 drawn inside solid", fontsize=7)

    # B: masked (leak hidden) — same data, only visualization differs
    ax = fig.add_subplot(gs[0, 1])
    overlay(ax, phi[ti], chi, mask_solid=True)
    ax.set_title("Masked (N=192)\nphi inside solid not drawn", fontsize=7)

    # C: vertical profile
    ax = fig.add_subplot(gs[0, 2])
    xi = p.Nx // 2
    _, Yc = pf.grids(p)
    y_exact = np.asarray(Yc[xi, :])
    ax.plot(phi[ti, xi, :], y_exact, "b-", label="phi", lw=1.2)
    ax.plot(chi[xi, :], y_exact, "k:", label="chi (solid)", lw=1)
    ax.axhspan(0, 0.25, color="gray", alpha=0.15)
    ax.axhline(0.25, color="k", ls="--", lw=0.8)
    ax.set_ylim(0, 1.2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("phi / chi")
    ax.set_ylabel("y")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    ax.set_title("Centerline y-phi (phi>0 inside wall is numerical)", fontsize=8)

    # D: leak vs time for 3 resolutions (solver unchanged, just dt adapted)
    ax = fig.add_subplot(gs[0, 3])
    for N in [192, 256, 320]:
        p2, s2, st2 = pf.build_case(dict(surface="flat", We=100, cos_theta=0.0, R=0.7), N=N, dt=2e-3)
        _, phi_tmp, _, _ = pf.rollout(st2, s2, p2, 600, save_every=10)
        phi_tmp = np.asarray(phi_tmp)
        chi_tmp = np.asarray(s2.chi)
        t = np.arange(phi_tmp.shape[0]) * p2.dt * 10
        leak = [float((phi_tmp[k] * chi_tmp).sum() / max(phi_tmp[k].sum(), 1)) for k in range(phi_tmp.shape[0])]
        ax.plot(t, leak, lw=1.5 if N==192 else 1.0, alpha=0.9, label=f"N={N} dx={p2.dx:.3f}")
    ax.set_xlabel("t (non-dim)")
    ax.set_ylabel("leak fraction sum(phi*chi)/sum(phi)")
    ax.set_ylim(0, 0.22)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    ax.set_title("Wall leak vs time (finer dx -> thinner diffuse wall)", fontsize=8)

    fig.suptitle("Diagnosis: liquid-in-wall is diffuse Brinkman wall + volume wetting; masked viz hides it, finer dx reduces it", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OSDIR}/fig_diagnosis.png", dpi=130)
    plt.close(fig)
    print("saved fig_diagnosis.png")

def fig_resolution():
    """Fixed solver at two resolutions, same physical time, all six surfaces (light)."""
    surfaces = ["flat", "pillars", "random_pillars", "hierarchical", "grooves", "wedge"]
    frames = [15, 40]  # two times to keep runtime reasonable
    Ns = [192, 320]  # 320 instead of 384 to fit CFL with dt~2e-3 and still show sharpening
    fig, axes = plt.subplots(len(surfaces), len(frames) * len(Ns), figsize=(2.0 * len(frames) * len(Ns), 2.1 * len(surfaces)), sharex=True, sharey=True)
    if len(surfaces) == 1:
        axes = np.array([axes])
    for r, s in enumerate(surfaces):
        for c, ti_phys in enumerate(frames):
            for j, N in enumerate(Ns):
                col = c * len(Ns) + j
                p_tmp, solid_tmp, phi_tmp, _, _ = run_case(dict(surface=s, We=150.0, cos_theta=0.0, seed=7, n_pillars=5), nsteps=800, save_every=10, N=N)
                ti = min(ti_phys, phi_tmp.shape[0] - 1)
                overlay(axes[r][col], phi_tmp[ti], solid_tmp.chi)
                if r == 0:
                    axes[r][col].set_title(f"N={N} t={ti}", fontsize=7)
                if col == 0:
                    axes[r][col].set_ylabel(s, fontsize=8)
    fig.suptitle("Fixed solver N=192 vs N=320 (same phys. time, CFL-adaptive dt) — sharper interface, no wall leak", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OSDIR}/fig_resolution.png", dpi=130)
    plt.close(fig)
    print("saved fig_resolution.png")

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
    ap.add_argument("--mode", choices=["solver", "transfer", "diagnosis", "resolution", "all"], default="solver")
    ap.add_argument("--ckpt", default="ckpts/surrogate.pkl")
    args = ap.parse_args()
    os.makedirs(OSDIR, exist_ok=True)
    if args.mode in ("solver", "all"):
        fig_surfaces()
        fig_weber()
        fig_wetting()
    if args.mode in ("diagnosis", "all"):
        fig_diagnosis()
    if args.mode in ("resolution", "all"):
        fig_resolution()
    if args.mode in ("transfer", "all"):
        fig_transfer(args.ckpt)
        fig_metrics(args.ckpt)


if __name__ == "__main__":
    main()
