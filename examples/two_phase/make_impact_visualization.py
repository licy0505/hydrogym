"""Generate comprehensive multi-surface droplet descent, impact, and spreading visualizations.

Tests 6 distinct solid surfaces:
1. flat (Smooth flat wall)
2. pillars (Periodic micro-pillar array)
3. grooves (Deep grooves / trenches)
4. hierarchical (Hierarchical dual-scale micro-texture)
5. random_pillars (Disordered roughness pillars)
6. wedge (Slanted asymmetric wedge)

Solves the floating droplet artifact by:
- Using dynamic impact momentum (u_impact = 1.6, We = 200) to overcome the air squeeze film.
- Refining the diffuse-interface visualization to eliminate threshold white gaps.
- Tracing the complete physical lifecycle from in-flight descent to steady wetted spreading.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import phasefield as pf
from PIL import Image

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_frame(ax, phi_k, solid, p, u_k=None, v_k=None, title=None, show_quivers=True):
    """Render solid and fluid phase field cleanly with zero artificial gap."""
    X, Y = pf.grids(p)
    extent = [0.0, float(p.Lx), 0.0, float(p.Ly)]

    # 1. Solid background (dark gray with crisp solid contour)
    ax.imshow(
        solid.chi.T,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        alpha=0.45,
    )
    if float(np.nanmin(solid.chi)) < 0.5 < float(np.nanmax(solid.chi)):
        ax.contour(X, Y, solid.chi, levels=[0.5], colors="#212121", linewidths=1.2)

    # 2. Liquid phase field (smooth Blues colormap down to phi=0.08, masked only inside solid core)
    mask = (phi_k < 0.08) | (solid.chi > 0.70)
    ax.imshow(
        np.ma.masked_where(mask.T, phi_k.T),
        origin="lower",
        extent=extent,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        alpha=0.90,
    )

    # 3. Bold droplet interface contour (phi = 0.5)
    if float(np.nanmin(phi_k)) < 0.5 < float(np.nanmax(phi_k)):
        ax.contour(X, Y, phi_k, levels=[0.5], colors="#0D47A1", linewidths=1.6)

    # 4. Velocity vectors
    if show_quivers and (u_k is not None) and (v_k is not None):
        sub = 6
        speed = np.sqrt(u_k**2 + v_k**2)
        q_mask = (phi_k > 0.15) & (speed > 0.06)
        qx = X[::sub, ::sub][q_mask[::sub, ::sub]]
        qy = Y[::sub, ::sub][q_mask[::sub, ::sub]]
        qu = u_k[::sub, ::sub][q_mask[::sub, ::sub]]
        qv = v_k[::sub, ::sub][q_mask[::sub, ::sub]]
        if len(qx) > 0:
            ax.quiver(
                qx,
                qy,
                qu,
                qv,
                color="#C62828",
                scale=28,
                width=0.005,
                headwidth=3.2,
                headlength=4.2,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8.5, fontweight="bold")


def run_simulation(case, n_steps=180, save_every=4, dt=2.0e-3):
    p, solid, st = pf.build_case(case, N=192, dt=dt)
    _, phi, u, v = pf.rollout(st, solid, p, n_steps, save_every=save_every)
    return p, solid, np.asarray(phi), np.asarray(u), np.asarray(v)


def compute_metrics(phi, u, v, solid, p):
    n_frames = phi.shape[0]
    X, Y = pf.grids(p)
    dx, dy = p.dx, p.dy
    surface_top = float(np.max(np.where(solid.sdf < 0.0, Y, -np.inf)))

    w_contacts, beta_ratios, ke_list = [], [], []
    r_initial = 0.7
    d0 = 2.0 * r_initial

    for k in range(n_frames):
        phi_k = phi[k]
        u_k = u[k]
        v_k = v[k]

        near_surface = (phi_k > 0.25) & (Y >= surface_top - 0.04) & (Y <= surface_top + 0.08)
        w_c = float(np.max(X[near_surface]) - np.min(X[near_surface])) if np.any(near_surface) else 0.0
        w_contacts.append(w_c)

        liq = phi_k > 0.4
        d_max = float(np.max(X[liq]) - np.min(X[liq])) if np.any(liq) else 0.0
        beta_ratios.append(d_max / d0)

        rho = 0.1 + 0.9 * phi_k
        ke = float(np.sum(0.5 * rho * (u_k**2 + v_k**2)) * dx * dy)
        ke_list.append(ke)

    return np.array(w_contacts), np.array(beta_ratios), np.array(ke_list), surface_top


def make_surfaces_matrix():
    """Figure 1: 6-surface complete descent-to-spreading comparison matrix."""
    print("Simulating descent and spreading across 6 distinct wall surfaces...")
    surfaces = [
        ("flat", "Flat Wall"),
        ("pillars", "Periodic Pillars"),
        ("grooves", "Micro-Grooves"),
        ("hierarchical", "Hierarchical Texture"),
        ("random_pillars", "Random Pillars"),
        ("wedge", "Asymmetric Wedge"),
    ]
    frame_indices = [0, 5, 12, 20, 30, 42]
    n_steps = 180
    save_every = 4
    dt = 2.0e-3
    times = [k * save_every * dt for k in frame_indices]

    fig, axes = plt.subplots(len(surfaces), len(frame_indices), figsize=(16, 13), dpi=140)

    for r, (s_name, s_label) in enumerate(surfaces):
        print(f"  Running surface: {s_name}...")
        case = dict(
            surface=s_name,
            We=200.0,
            Re=200.0,
            cos_theta=0.75,
            R=0.7,
            u_impact=1.6,
            seed=7,
            n_pillars=5,
            dt=dt,
        )
        p, solid, phi, u, v = run_simulation(case, n_steps=n_steps, save_every=save_every, dt=dt)
        w_contacts, betas, _, surf_top = compute_metrics(phi, u, v, solid, p)

        y_min_crop = max(0.12, surf_top - 0.25)
        y_max_crop = y_min_crop + 2.0

        for c, f_idx in enumerate(frame_indices):
            ax = axes[r, c]
            title = f"t = {times[c]:.3f} s" if r == 0 else None
            plot_frame(ax, phi[f_idx], solid, p, u[f_idx], v[f_idx], title=title)
            ax.set_xlim(1.2, 4.8)
            ax.set_ylim(y_min_crop, y_max_crop)

            if c == 0:
                ax.set_ylabel(f"{s_label}", fontsize=9.5, fontweight="bold")

            # Footprint annotation
            ax.text(
                0.04,
                0.90,
                f"$w_c$={w_contacts[f_idx]:.2f}",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85),
                verticalalignment="top",
            )

    fig.suptitle(
        "Direct Numerical Simulation: Droplet Descent, Wall Impact, and Spreading across 6 Solid Geometries\n"
        "(We = 200, Re = 200, Impact Velocity $u_0 = 1.6$, Hydrophilic Contact Angle $\\theta_Y \\approx 41^\\circ$)",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(FIG_DIR, "droplet_impact_surfaces.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_lifecycle_process():
    """Figure 2: Detailed 6-stage lifecycle on micro-pillars with diagnostic curves."""
    print("Simulating high-resolution lifecycle process on micro-pillars...")
    case = dict(
        surface="pillars",
        n_pillars=6,
        width=0.35,
        height=0.35,
        cos_theta=0.80,
        We=220.0,
        Re=220.0,
        R=0.7,
        u_impact=1.7,
        dt=1.8e-3,
    )
    n_steps = 220
    save_every = 5
    dt = 1.8e-3
    p, solid, phi, u, v = run_simulation(case, n_steps=n_steps, save_every=save_every, dt=dt)

    times = np.arange(len(phi)) * save_every * dt
    w_contacts, betas, ke, surf_top = compute_metrics(phi, u, v, solid, p)

    stage_indices = [0, 5, 12, 22, 32, 42]
    stage_titles = [
        "(1) In-Flight Descent\nt = 0.000 s",
        "(2) Wall Touchdown\nt = 0.045 s",
        "(3) Radial Jetting & Spreading\nt = 0.108 s",
        "(4) Max Spreading & Imbibition\nt = 0.198 s",
        "(5) Capillary Retraction\nt = 0.288 s",
        "(6) Equilibrium Wetted State\nt = 0.378 s",
    ]

    fig = plt.figure(figsize=(16, 11), dpi=150)
    gs = fig.add_gridspec(3, 6, height_ratios=[1.2, 1.2, 0.9], hspace=0.36, wspace=0.28)

    y_crop = (0.16, 2.2)
    x_crop = (1.2, 4.8)

    for i, idx in enumerate(stage_indices):
        row = i // 3
        col = (i % 3) * 2
        ax = fig.add_subplot(gs[row, col : col + 2])
        plot_frame(ax, phi[idx], solid, p, u[idx], v[idx], title=stage_titles[i])
        ax.set_xlim(x_crop)
        ax.set_ylim(y_crop)
        ax.set_xlabel("x (dimensionless)", fontsize=9)
        ax.set_ylabel("y (dimensionless)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.45)

        ax.text(
            0.04,
            0.92,
            f"Wetted $w_c$={w_contacts[idx]:.2f}\nSpread $\\beta$={betas[idx]:.2f}",
            transform=ax.transAxes,
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
            verticalalignment="top",
        )

    # Diagnostic 1: Wetted Contact Footprint
    ax_d1 = fig.add_subplot(gs[2, 0:2])
    ax_d1.plot(times, w_contacts, color="#0D47A1", lw=2.2, label="Wetted Width $w_c(t)$")
    for idx in stage_indices:
        ax_d1.axvline(times[idx], color="gray", linestyle="--", alpha=0.4)
    ax_d1.set_xlabel("Time $t$ [s]", fontsize=10)
    ax_d1.set_ylabel("Contact Footprint Width $w_c$", fontsize=10)
    ax_d1.set_title("Dynamic Contact Line Spreading", fontsize=11, fontweight="bold")
    ax_d1.grid(True, linestyle=":", alpha=0.6)
    ax_d1.legend(loc="lower right", fontsize=8.5)

    # Diagnostic 2: Dimensionless Spreading Factor beta(t)
    ax_d2 = fig.add_subplot(gs[2, 2:4])
    ax_d2.plot(times, betas, color="#2E7D32", lw=2.2, label="Spreading Factor $\\beta = D(t)/D_0$")
    ax_d2.axhline(1.0, color="black", linestyle=":", label="Initial Droplet $D_0$")
    for idx in stage_indices:
        ax_d2.axvline(times[idx], color="gray", linestyle="--", alpha=0.4)
    ax_d2.set_xlabel("Time $t$ [s]", fontsize=10)
    ax_d2.set_ylabel("$\\beta = D(t) / D_0$", fontsize=10)
    ax_d2.set_title("Dimensionless Droplet Deformation", fontsize=11, fontweight="bold")
    ax_d2.grid(True, linestyle=":", alpha=0.6)
    ax_d2.legend(loc="upper right", fontsize=8.5)

    # Diagnostic 3: Kinetic Energy Dissipation
    ax_d3 = fig.add_subplot(gs[2, 4:6])
    ax_d3.plot(times, ke, color="#C62828", lw=2.2, label="Kinetic Energy $E_k(t)$")
    for idx in stage_indices:
        ax_d3.axvline(times[idx], color="gray", linestyle="--", alpha=0.4)
    ax_d3.set_xlabel("Time $t$ [s]", fontsize=10)
    ax_d3.set_ylabel("Total Kinetic Energy $E_k$", fontsize=10)
    ax_d3.set_title("Impact Kinetic Energy Dissipation", fontsize=11, fontweight="bold")
    ax_d3.grid(True, linestyle=":", alpha=0.6)
    ax_d3.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Complete Physical Lifecycle: Droplet Descent, Touchdown, Lamella Jetting, and Texture Imbibition\n"
        "(We = 220, Re = 220, Impact Velocity $u_0 = 1.7$, Contact Angle $\\theta_Y \\approx 36^\\circ$)",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )

    out_path = os.path.join(FIG_DIR, "droplet_impact_process.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_animation_gif():
    """Figure 3: Smooth 15 FPS animation of descent and spreading."""
    print("Generating animated GIF of descent and spreading...")
    case = dict(
        surface="pillars",
        n_pillars=6,
        width=0.35,
        height=0.35,
        cos_theta=0.80,
        We=220.0,
        Re=220.0,
        R=0.7,
        u_impact=1.7,
        dt=1.8e-3,
    )
    n_steps = 220
    save_every = 5
    dt = 1.8e-3
    p, solid, phi, u, v = run_simulation(case, n_steps=n_steps, save_every=save_every, dt=dt)
    times = np.arange(len(phi)) * save_every * dt
    w_contacts, betas, ke, _ = compute_metrics(phi, u, v, solid, p)

    tmp_frame_dir = "figures/tmp_frames"
    os.makedirs(tmp_frame_dir, exist_ok=True)
    frames = []

    fig, (ax_main, ax_diag) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=110, gridspec_kw={"width_ratios": [1.4, 1.0]})

    for k in range(len(phi)):
        ax_main.clear()
        ax_diag.clear()

        plot_frame(
            ax_main,
            phi[k],
            solid,
            p,
            u[k],
            v[k],
            title=f"Descent & Spreading | t = {times[k]:.3f} s",
        )
        ax_main.set_xlim(1.2, 4.8)
        ax_main.set_ylim(0.16, 2.2)
        ax_main.grid(True, linestyle=":", alpha=0.45)

        ax_main.text(
            0.04,
            0.92,
            f"Contact Width $w_c$: {w_contacts[k]:.2f}\nSpreading Factor $\\beta$: {betas[k]:.2f}",
            transform=ax_main.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
            verticalalignment="top",
        )

        ax_diag.plot(times[: k + 1], w_contacts[: k + 1], color="#0D47A1", lw=2.2, label="Wetted Width $w_c(t)$")
        ax_diag.plot(
            times[: k + 1],
            betas[: k + 1],
            color="#2E7D32",
            lw=2.0,
            linestyle="--",
            label="Spreading Factor $\\beta(t)$",
        )
        ax_diag.scatter([times[k]], [w_contacts[k]], color="#0D47A1", s=40, zorder=5)
        ax_diag.scatter([times[k]], [betas[k]], color="#2E7D32", s=40, zorder=5)

        ax_diag.set_xlim(0.0, times[-1])
        ax_diag.set_ylim(0.0, max(np.max(w_contacts), np.max(betas)) * 1.15)
        ax_diag.set_xlabel("Time $t$ [s]", fontsize=9.5)
        ax_diag.set_ylabel("Metric Value", fontsize=9.5)
        ax_diag.set_title("Real-Time Spreading Diagnostics", fontsize=11, fontweight="bold")
        ax_diag.grid(True, linestyle=":", alpha=0.6)
        ax_diag.legend(loc="upper left", fontsize=8.5)

        frame_file = os.path.join(tmp_frame_dir, f"frame_{k:03d}.png")
        fig.savefig(frame_file, bbox_inches="tight")
        frames.append(Image.open(frame_file))

    plt.close(fig)

    gif_path = os.path.join(FIG_DIR, "droplet_impact_animation.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=65,  # ~15 fps
        loop=0,
    )
    print(f"Saved {gif_path}")

    for f in os.listdir(tmp_frame_dir):
        os.remove(os.path.join(tmp_frame_dir, f))
    os.rmdir(tmp_frame_dir)


if __name__ == "__main__":
    make_surfaces_matrix()
    make_lifecycle_process()
    make_animation_gif()
    print("All multi-surface visualizations successfully generated!")
