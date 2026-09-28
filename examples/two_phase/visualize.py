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

import matplotlib
import numpy as np

matplotlib.use("Agg")
import cases as C
import matplotlib.pyplot as plt
import phasefield as pf

OSDIR = "figures"


def flip(a):
    """Orient arrays with the wall at the bottom for display."""
    return np.flipud(np.asarray(a).T)


def overlay(
    ax,
    phi,
    chi,
    title=None,
    truth=None,
    mask_solid=True,
    display_thresh=0.10,
    truth_levels=(0.5,),
):
    # Solid background with crisp contour to eliminate floating gap
    ax.imshow(flip(chi), cmap="gray_r", vmin=0, vmax=1, origin="lower")
    chi_f = flip(chi)
    if float(np.nanmin(chi_f)) < 0.5 < float(np.nanmax(chi_f)):
        ax.contour(chi_f, levels=[0.5], colors="#212121", linewidths=1.0)

    if mask_solid:
        mask = (flip(phi) < display_thresh) | (flip(chi) > 0.65)
    else:
        mask = flip(phi) < display_thresh
    ax.imshow(np.ma.masked_where(mask, flip(phi)), cmap="Blues", vmin=0, vmax=1, alpha=0.92)

    # Liquid interface contour
    phi_f = flip(phi)
    if float(np.nanmin(phi_f)) < 0.5 < float(np.nanmax(phi_f)):
        ax.contour(phi_f, levels=[0.5], colors="#0D47A1", linewidths=1.2)

    if truth is not None:
        if mask_solid:
            truth = np.where(np.asarray(chi) > 0.5, 0.0, truth)
        truth_f = flip(truth)
        for level in truth_levels:
            if float(np.nanmin(truth_f)) <= level <= float(np.nanmax(truth_f)):
                ax.contour(truth_f, levels=[level], colors="r", linewidths=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def run_case(case, nsteps=1200, save_every=10, N=192, dt=None):
    dt_eff = case.get("dt", dt if dt is not None else 4e-3)
    p, solid, st = pf.build_case(case, N=N, dt=dt_eff)
    # low-We may need longer nsteps; caller can pass larger nsteps
    _, phi, u, v = pf.rollout(st, solid, p, nsteps, save_every=save_every)
    return p, solid, np.asarray(phi), np.asarray(u), np.asarray(v)


# -------------------------------------------------------------------------------------
#  solver figures
# -------------------------------------------------------------------------------------


def fig_surfaces():
    surfaces = ["flat", "pillars", "random_pillars", "hierarchical", "grooves", "wedge"]
    frames = [0, 6, 14, 24, 38]
    fig, axes = plt.subplots(len(surfaces), len(frames), figsize=(2.4 * len(frames), 2.3 * len(surfaces)))
    for r, s in enumerate(surfaces):
        case = dict(
            surface=s,
            We=200.0,
            Re=200.0,
            cos_theta=0.75,
            u_impact=1.6,
            seed=7,
            n_pillars=5,
            dt=2.5e-3,
        )
        p, solid, phi, _, _ = run_case(case, nsteps=160, save_every=4, dt=2.5e-3)
        T = phi.shape[0]
        for c, ti in enumerate(frames):
            ti = min(ti, T - 1)
            phys_t = (ti + 1) * 4 * p.dt if ti > 0 else 0.0
            title = f"t = {phys_t:.2f} s" if r == 0 else None
            overlay(axes[r][c], phi[ti], solid.chi, title=title)
            if c == 0:
                axes[r][c].set_ylabel(s, fontsize=9.5, fontweight="bold")
    fig.suptitle(
        "Droplet Descent & Spreading across Surface Families (We=200, u0=1.6, cos $\\theta$=0.75)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(f"{OSDIR}/fig_surfaces.png", dpi=120)
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
    cts = [-0.85, 0.0, 0.85]
    frames = [0, 6, 14, 24, 38]
    fig, axes = plt.subplots(len(cts), len(frames), figsize=(2.4 * len(frames), 2.3 * len(cts)))
    labels = ["Hydrophobic\n$\\cos\\theta$=-0.85", "Neutral\n$\\cos\\theta$=0.0", "Hydrophilic\n$\\cos\\theta$=+0.85"]
    for r, ct in enumerate(cts):
        case = dict(surface="flat", We=200.0, Re=200.0, cos_theta=ct, u_impact=1.6, dt=2.5e-3)
        p, solid, phi, _, _ = run_case(case, nsteps=160, save_every=4, dt=2.5e-3)
        T = phi.shape[0]
        for c, ti in enumerate(frames):
            ti = min(ti, T - 1)
            phys_t = (ti + 1) * 4 * p.dt if ti > 0 else 0.0
            title = f"t = {phys_t:.2f} s" if r == 0 else None
            overlay(axes[r][c], phi[ti], solid.chi, title=title)
            if c == 0:
                axes[r][c].set_ylabel(labels[r], fontsize=9)
    fig.suptitle("Wettability Sweep: Hydrophobic Rebound vs Hydrophilic Spreading (We=200)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OSDIR}/fig_wetting.png", dpi=110)
    plt.close(fig)
    print("saved fig_wetting.png")


def fig_lowWe():
    """Low-We sweep using the same guarded cases as the training generator.

    The old figure intentionally started the drop inside the wall and used the
    high-We resolution/viscosity settings.  It therefore displayed numerical
    spikes rather than a low-We deposition sequence.  Frame selection below is
    done in physical time, not by an arbitrary frame number.
    """
    Wes = [10, 18, 32, 55]
    frame_times = [0.2, 0.6, 1.0, 1.4]
    fig, axes = plt.subplots(len(Wes), len(frame_times), figsize=(2.2 * len(frame_times), 2.3 * len(Wes)))
    for r, we in enumerate(Wes):
        # Match cases._low_common: lower Re, a better-resolved interface, a
        # small wetting affinity, and a positive gap above the wall.
        u = 0.18 if we < 15 else 0.22 if we < 35 else 0.28
        Re = float(np.clip(200.0 * u / 0.5, 60.0, 120.0))
        dt = 1e-3 if we < 20 else 2e-3
        nsteps = 2500 if we < 20 else 1800
        case = dict(
            surface="flat",
            We=we,
            Re=Re,
            cos_theta=0.0,
            u_impact=u,
            R=0.65,
            eps_factor=3.0,
            wall_energy_amp=0.5,
            wet_band=0.08,
            impact_gap_eps=2.5,
            velocity_mode="streamfunction",
            dt=dt,
        )
        p, solid, phi, _, _ = run_case(case, nsteps=nsteps, save_every=10, N=192)
        T = phi.shape[0]
        frame_dt = p.dt * 10.0
        for c, time_target in enumerate(frame_times):
            ti = min(max(int(round(time_target / frame_dt)) - 1, 0), T - 1)
            overlay(axes[r][c], phi[ti], solid.chi, title=f"t={((ti + 1) * frame_dt):.2f}" if r == 0 else None)
            if c == 0:
                axes[r][c].set_ylabel(f"We={we}\nRe={Re:.0f}", fontsize=9)
    fig.suptitle("Low-We sweep with guarded initialization and physical-time frames", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OSDIR}/fig_lowWe.png", dpi=110)
    plt.close(fig)
    print("saved fig_lowWe.png")


def fig_lowWe_transfer(ckpt="ckpts/lowWe_fno_sdf_u3.pkl", data_dir="data/lowWe_all"):
    """Full-horizon low-We transfer: every unseen family, median + worst."""
    import evaluate_transfer as E
    import surrogate as S

    predict, _, geom_mode = E.make_predictor(ckpt)
    tests = S.load_full(data_dir, "test")
    expected = ("random_pillars", "hierarchical", "grooves", "wedge")
    grouped = {fam: [c for c in tests if c["surface"] == fam] for fam in expected}
    missing = [fam for fam, cs in grouped.items() if not cs]
    if missing:
        raise RuntimeError(f"missing low-We TEST families in {data_dir}: {missing}")

    def rollout_case(c):
        geom = S.geometry_features(c["chi"], c["sdf"], float(c["scalars"][3]), geom_mode)
        cur = np.stack([c["phi"][0], c["u"][0], c["v"][0]], -1)
        traj = [cur[..., 0].copy()]
        raw_m, proj_l1 = [], []
        for _ in range(c["phi"].shape[0] - 1):
            cur_b, aux = predict(cur[None], geom[None], c["scalars"][None], return_aux=True)
            cur = cur_b[0]
            traj.append(cur[..., 0].copy())
            raw_m.append(float(aux["raw_mass_rel"][0]))
            proj_l1.append(float(aux["projection_l1_rel"][0]))
        return np.stack(traj), float(np.mean(raw_m)), float(np.mean(proj_l1))

    rows = []
    for fam in expected:
        scored = []
        for c in grouped[fam]:
            traj, raw_m, proj_l1 = rollout_case(c)
            final_iou = E.iou(traj[-1], c["phi"][-1])
            scored.append((final_iou, c, traj, raw_m, proj_l1))
        scored.sort(key=lambda x: x[0])
        for rank, item in (("worst", scored[0]), ("median", scored[len(scored) // 2])):
            rows.append((fam, rank, *item))

    fig, axes = plt.subplots(len(rows), 5, figsize=(11.5, 2.15 * len(rows)))
    for r, (fam, rank, final_iou, c, traj, raw_m, proj_l1) in enumerate(rows):
        T = min(traj.shape[0], c["phi"].shape[0])
        frames = np.rint(np.linspace(0, T - 1, 5)).astype(int)
        fluid = (c["sdf"] >= 0.0).astype(np.float32)
        m0 = max(float(np.sum(c["phi"][0] * fluid)), 1e-12)
        for ci, ti in enumerate(frames):
            pred_mass = float(np.sum(traj[ti] * fluid) / m0)
            truth_mass = float(np.sum(c["phi"][ti] * fluid) / m0)
            overlay(
                axes[r][ci],
                traj[ti],
                c["chi"],
                truth=c["phi"][ti],
                display_thresh=0.05,
                truth_levels=(0.1, 0.5),
            )
            axes[r][ci].text(
                0.02,
                0.02,
                f"M/M0 {pred_mass:.3f}/{truth_mass:.3f}",
                transform=axes[r][ci].transAxes,
                fontsize=5.2,
                va="bottom",
            )
            axes[r][ci].set_title(f"t={float(c['time'][ti]):.3f}", fontsize=7)
            if ci == 0:
                axes[r][ci].set_ylabel(
                    f"{fam} · {rank}\nWe={float(c['scalars'][0]) * 100:.0f}\n"
                    f"IoUfinal={final_iou:.3f}\nrawM={raw_m:.1e} projL1={proj_l1:.1e}",
                    fontsize=6.5,
                )

    fig.suptitle(
        "Low-We full-horizon transfer — every unseen family; median and worst case",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(f"{OSDIR}/fig_lowWe_transfer.png", dpi=150)
    plt.close(fig)
    print("saved fig_lowWe_transfer.png")


def fig_diagnosis():
    """Ablation: reproduce the legacy wall leak only with solid projection disabled."""
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.2, 1.6, 1.6], wspace=0.35)

    case = dict(surface="flat", We=100.0, cos_theta=0.0, R=0.7, enforce_solid_phi=False)

    # Deliberately disable the current mass-conserving solid projection.
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
        p2, s2, st2 = pf.build_case(
            dict(surface="flat", We=100, cos_theta=0.0, R=0.7, enforce_solid_phi=False),
            N=N,
            dt=2e-3,
        )
        _, phi_tmp, _, _ = pf.rollout(st2, s2, p2, 600, save_every=10)
        phi_tmp = np.asarray(phi_tmp)
        chi_tmp = np.asarray(s2.chi)
        t = np.arange(phi_tmp.shape[0]) * p2.dt * 10
        leak = [float((phi_tmp[k] * chi_tmp).sum() / max(phi_tmp[k].sum(), 1)) for k in range(phi_tmp.shape[0])]
        ax.plot(t, leak, lw=1.5 if N == 192 else 1.0, alpha=0.9, label=f"N={N} dx={p2.dx:.3f}")
    ax.set_xlabel("t (non-dim)")
    ax.set_ylabel("leak fraction sum(phi*chi)/sum(phi)")
    ax.set_ylim(0, 0.22)
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    ax.set_title("Wall leak vs time (finer dx -> thinner diffuse wall)", fontsize=8)

    fig.suptitle(
        "Diagnosis: liquid-in-wall is diffuse Brinkman wall + volume wetting; masked viz hides it, finer dx reduces it",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OSDIR}/fig_diagnosis.png", dpi=130)
    plt.close(fig)
    print("saved fig_diagnosis.png")


def fig_resolution():
    """Compare resolutions at identical physical times, not equal frame indices."""
    surfaces = ["flat", "pillars", "random_pillars", "hierarchical", "grooves", "wedge"]
    target_times = [0.6, 1.4]
    Ns = [192, 320]
    fig, axes = plt.subplots(
        len(surfaces),
        len(target_times) * len(Ns),
        figsize=(2.0 * len(target_times) * len(Ns), 2.1 * len(surfaces)),
        sharex=True,
        sharey=True,
    )
    if len(surfaces) == 1:
        axes = np.array([axes])
    for r, s in enumerate(surfaces):
        for c, time_target in enumerate(target_times):
            for j, N in enumerate(Ns):
                col = c * len(Ns) + j
                save_every = 10
                p_tmp, solid_tmp, phi_tmp, _, _ = run_case(
                    dict(surface=s, We=150.0, cos_theta=0.0, seed=7, n_pillars=5),
                    nsteps=800,
                    save_every=save_every,
                    N=N,
                )
                frame_dt = p_tmp.dt * save_every
                ti = min(max(int(round(time_target / frame_dt)) - 1, 0), phi_tmp.shape[0] - 1)
                actual_time = (ti + 1) * frame_dt
                overlay(axes[r][col], phi_tmp[ti], solid_tmp.chi)
                if r == 0:
                    axes[r][col].set_title(f"N={N} t={actual_time:.3f}", fontsize=7)
                if col == 0:
                    axes[r][col].set_ylabel(s, fontsize=8)
    fig.suptitle(
        "Fixed solver N=192 vs N=320 (same phys. time, CFL-adaptive dt) — sharper interface, no wall leak", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(f"{OSDIR}/fig_resolution.png", dpi=130)
    plt.close(fig)
    print("saved fig_resolution.png")


# -------------------------------------------------------------------------------------
#  transfer figures
# -------------------------------------------------------------------------------------


def _rollout_surrogate_current(ckpt, c, K=None):
    import evaluate_transfer as E
    import surrogate as S

    predict, _, geom_mode = E.make_predictor(ckpt)
    geom = S.geometry_features(c["chi"], c["sdf"], float(c["scalars"][3]), geom_mode)
    if K is None:
        K = c["phi"].shape[0]
    K = min(K, c["phi"].shape[0])
    st = np.stack([c["phi"][0], c["u"][0], c["v"][0]], -1)
    out = [st[..., 0].copy()]
    for _ in range(K - 1):
        st = predict(st[None], geom[None], c["scalars"][None])[0]
        out.append(st[..., 0].copy())
    return np.stack(out)


def fig_transfer(ckpt, data_dir="data/base"):
    import surrogate as S

    train = S.load_full(data_dir, "train")
    test = S.load_full(data_dir, "test")
    picks = [c for c in train if c["surface"] == "flat"][:1]
    for fam in ("random_pillars", "hierarchical", "grooves", "wedge"):
        picks += [c for c in test if c["surface"] == fam][:1]
    if not picks:
        raise RuntimeError(f"no transfer trajectories found in {data_dir}")

    fig, axes = plt.subplots(len(picks), 5, figsize=(11.0, 2.25 * len(picks)))
    if len(picks) == 1:
        axes = axes[None]
    for r, c in enumerate(picks):
        pred = _rollout_surrogate_current(ckpt, c)
        T = min(len(pred), c["phi"].shape[0])
        frames = np.rint(np.linspace(0, T - 1, 5)).astype(int)
        for ci, ti in enumerate(frames):
            overlay(axes[r][ci], pred[ti], c["chi"], truth=c["phi"][ti])
            axes[r][ci].set_title(f"t={float(c['time'][ti]):.3f}", fontsize=7)
            if ci == 0:
                axes[r][ci].set_ylabel(c["surface"], fontsize=8)
    fig.suptitle("Full-horizon surrogate rollout (blue) vs solver truth (red)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OSDIR}/fig_transfer.png", dpi=130)
    plt.close(fig)
    print("saved fig_transfer.png")


def spreading(phi, L=6.0):
    m = phi > 0.5
    if not m.any():
        return 0.0
    xs = np.where(m.any(axis=1))[0]
    return (xs.max() - xs.min()) * (L / phi.shape[0])


def fig_metrics(ckpt, data_dir="data/base"):
    import surrogate as S

    tests = S.load_full(data_dir, "test")
    picks = []
    for fam in ("random_pillars", "hierarchical", "grooves", "wedge"):
        picks += [c for c in tests if c["surface"] == fam][:1]
    fig, axes = plt.subplots(2, len(picks), figsize=(3.5 * len(picks), 6))
    if len(picks) == 1:
        axes = axes[:, None]
    for i, c in enumerate(picks):
        pred = _rollout_surrogate_current(ckpt, c)
        K = min(len(pred), c["phi"].shape[0])
        tt = c["time"][:K]
        d_true = [spreading(c["phi"][t]) for t in range(K)]
        d_pred = [spreading(pred[t]) for t in range(K)]
        fluid = (c["sdf"] >= 0.0).astype(np.float32)
        m0 = max(float(np.sum(c["phi"][0] * fluid)), 1e-12)
        m_true = [float(np.sum(c["phi"][t] * fluid) / m0) for t in range(K)]
        m_pred = [float(np.sum(pred[t] * fluid) / m0) for t in range(K)]
        axes[0][i].plot(tt, d_true, "r-", label="truth")
        axes[0][i].plot(tt, d_pred, "b--", label="surrogate")
        axes[0][i].set_title(f"D(t) {c['surface']}", fontsize=9)
        axes[1][i].plot(tt, m_true, "r-")
        axes[1][i].plot(tt, m_pred, "b--")
        axes[1][i].set_title(f"fluid M/M0 {c['surface']}", fontsize=9)
        axes[1][i].set_xlabel("physical time")
    if picks:
        axes[0][0].legend(fontsize=7)
    fig.suptitle("Full-horizon transfer diagnostics on every unseen surface family", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OSDIR}/fig_metrics.png", dpi=130)
    plt.close(fig)
    print("saved fig_metrics.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["solver", "transfer", "diagnosis", "resolution", "lowWe", "all"], default="solver"
    )
    ap.add_argument("--ckpt", default="ckpts/surrogate.pkl")
    ap.add_argument("--data", default="data/base")
    ap.add_argument("--lowwe-ckpt", default="ckpts/lowWe_fno_sdf_u3.pkl")
    ap.add_argument("--lowwe-data", default="data/lowWe_all")
    args = ap.parse_args()
    os.makedirs(OSDIR, exist_ok=True)
    if args.mode in ("solver", "all"):
        fig_surfaces()
        fig_weber()
        fig_wetting()
        fig_lowWe()
    if args.mode in ("diagnosis", "all"):
        fig_diagnosis()
    if args.mode in ("resolution", "all"):
        fig_resolution()
    if args.mode in ("transfer", "all"):
        fig_transfer(args.ckpt, args.data)
        fig_metrics(args.ckpt, args.data)
    if args.mode == "lowWe":
        fig_lowWe()
    if args.mode in ("lowWe", "all"):
        fig_lowWe_transfer(args.lowwe_ckpt, args.lowwe_data)


if __name__ == "__main__":
    main()
