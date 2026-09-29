"""
Near-wall three-phase (liquid / gas / solid) contact-line close-up figures.

After a high-inertia impact the droplet is driven against a structured wall, and
the physics that actually controls the wetted footprint happens in a layer only a
few cells thick: the **three-phase contact line**, where

  * liquid      (phi > 0.5, outside the solid),
  * gas         (phi < 0.5, outside the solid),
  * solid       (chi > 0.5)

meet.  A full-domain plot of a 192^2 field cannot resolve it, so this script
re-runs the solver and renders (a) the full field, (b) a near-wall close-up and
(c) a cell-level zoom of the triple point, plus the vertical phi/chi profile
through the contact line and the time history of the wetted observables.

Everything here uses the *solver* only -- no surrogate checkpoint is needed.

Usage::

    python make_contact_closeup.py                 # default pillars case
    python make_contact_closeup.py --surface flat  # flat wall close-up
    python make_contact_closeup.py --steps 1600 --save-every 10
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import phasefield as pf
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

FIG_DIR = "figures"

# Three-phase categorical palette: gas / liquid / solid.
PHASE_CMAP = ListedColormap(["#f7fbff", "#9ecae1", "#4a4a4a"])
PHASE_LABELS = ["gas", "liquid", "solid"]


def _phase_index(phi, chi, level=0.5):
    """Map (phi, chi) to categorical indices: 0 = gas, 1 = liquid, 2 = solid."""
    solid = np.asarray(chi) > level
    liquid = (~solid) & (np.asarray(phi) > level)
    out = np.zeros(np.shape(phi), dtype=np.int32)
    out[liquid] = 1
    out[solid] = 2
    return out


def _solid_top(sdf, Y):
    """Highest solid point of every column; -inf where a column has no solid."""
    solid = np.asarray(sdf) < 0.0
    return np.where(solid, np.asarray(Y), -np.inf).max(axis=1)


def _wetted_columns(phi, sdf, Y, level=0.5, band=0.15):
    """Columns whose *wetting band* -- the same |sdf| < band layer the solver's
    ``contact_area`` observable uses -- actually contains liquid.

    The band cannot by itself separate "in flight" from "touching": the drop is
    released only ``2*eps`` above the surface, and the Brinkman wall plus the
    diffuse interface depress phi by ~2.5*eps next to the solid.  Touchdown is
    therefore decided separately, from the descent history (see
    :func:`touchdown_index`).
    """
    Y = np.asarray(Y)
    wet = (np.asarray(phi) > level) & (np.asarray(sdf) >= 0.0) & (np.asarray(sdf) < band)
    has_solid = np.isfinite(_solid_top(sdf, Y))
    return np.where(wet.any(axis=1) & has_solid)[0]


def touchdown_index(phi, Y, level=0.5, frac=0.5):
    """First frame in which the drop has completed ``frac`` of its total descent.

    The drop is released only ``2*eps`` above the surface and the impact is
    strongly overdamped, so the whole descent happens within a few frames of the
    initial clearance.  Judging touchdown from the *fraction of the total
    descent* is therefore both scale-free and robust, and -- unlike a fixed
    distance threshold -- cannot fire while the drop is still airborne.
    """
    y_rows = np.asarray(Y)[0, :]
    low = np.array(
        [float(y_rows[m.any(axis=0)].min()) if m.any() else np.nan for m in (np.asarray(f) > level for f in phi)]
    )
    fallen = low[0] - low
    total = float(np.nanmax(fallen))
    if not np.isfinite(total) or total <= 0.0:
        return None
    hit = np.where(fallen >= frac * total)[0]
    return int(hit[0]) if len(hit) else None


def find_contact_line(phi, sdf, X, Y, level=0.5, band=0.15, tip=0.35):
    """Locate the leading (left) three-phase contact line.

    Returns ``(x_cl, y_solid, y_free)``: the point where the wetted leading edge
    meets the solid, and the height of the free surface directly above it (the
    local wetting height).  ``None`` when no part of the drop is wetted yet.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    cols = _wetted_columns(phi, sdf, Y, level=level, band=band)
    if len(cols) == 0:
        return None
    i_cl = int(cols.min())
    x_cl = float(X[i_cl, 0])
    y_solid = float(_solid_top(sdf, Y)[i_cl])
    liquid_col = np.asarray(phi)[i_cl] > level
    cap = liquid_col & (Y[i_cl] >= y_solid) & (Y[i_cl] <= y_solid + tip)
    y_free = float(Y[i_cl][cap].max()) if cap.any() else y_solid
    return x_cl, y_solid, y_free


def _grid_dy(Y):
    """Vertical cell size.  ``pf.grids`` uses ``indexing='ij'``, so Y varies along axis 1."""
    Y = np.asarray(Y)
    return float(abs(Y[0, 1] - Y[0, 0]))


def wetted_width(phi, sdf, Y, level=0.5, band=0.15):
    """Horizontal extent of the wetted footprint."""
    cols = _wetted_columns(phi, sdf, Y, level=level, band=band)
    if len(cols) < 2:
        return 0.0
    return float(cols.max() - cols.min() + 1) * _grid_dy(Y)


def _plot_full(ax, phi, chi, p, box, title, wall_top):
    X, Y = pf.grids(p)
    extent = [0.0, float(p.Lx), 0.0, float(p.Ly)]
    mask = (phi < 0.08) | (chi > 0.70)
    ax.imshow(chi.T, origin="lower", extent=extent, cmap="gray_r", vmin=0.0, vmax=1.0, alpha=0.45)
    ax.imshow(np.ma.masked_where(mask.T, phi.T), origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=1, alpha=0.9)
    if float(np.nanmin(phi)) < 0.5 < float(np.nanmax(phi)):
        ax.contour(X, Y, phi, levels=[0.5], colors="#0D47A1", linewidths=1.1)
    if box is not None:
        x0, x1, y0, y1 = box
        ax.add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#D7263D", linewidth=1.6, linestyle="--")
        )
    ax.axhline(wall_top, color="#555555", linewidth=0.6, alpha=0.5)
    ax.set_xlim(1.2, 4.8)
    ax.set_ylim(max(0.0, wall_top - 0.35), wall_top + 2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold")


def _plot_closeup(ax, phi, chi, p, box, cl, wall_top, dz):
    """Three-phase near-wall close-up with the contact-line triple point marked."""
    x0, x1, y0, y1 = box
    idx = _phase_index(phi, chi)
    n_x, n_y = idx.shape
    # Cell-centred coordinates of the sub-window.
    xs = (np.arange(n_x) + 0.5) * p.dx
    ys = (np.arange(n_y) + 0.5) * p.dy
    sx = (xs >= x0) & (xs <= x1)
    sy = (ys >= y0) & (ys <= y1)
    extent = [
        float(xs[sx].min() - 0.5 * p.dx),
        float(xs[sx].max() + 0.5 * p.dx),
        float(ys[sy].min() - 0.5 * p.dy),
        float(ys[sy].max() + 0.5 * p.dy),
    ]

    im = ax.imshow(
        idx[np.ix_(sx, sy)].T,
        origin="lower",
        extent=extent,
        cmap=PHASE_CMAP,
        vmin=-0.5,
        vmax=2.5,
        interpolation="nearest",
    )
    Xw, Yw = np.meshgrid(xs[sx], ys[sy], indexing="ij")
    phiw = np.asarray(phi)[np.ix_(sx, sy)]
    chiw = np.asarray(chi)[np.ix_(sx, sy)]
    if float(phiw.min()) < 0.5 < float(phiw.max()):
        ax.contour(Xw, Yw, phiw, levels=[0.5], colors="#08306b", linewidths=1.5)
    if float(chiw.min()) < 0.5 < float(chiw.max()):
        ax.contour(Xw, Yw, chiw, levels=[0.5], colors="#111111", linewidths=1.2)

    if cl is not None:
        x_cl, y_solid, y_free = cl
        ax.plot([x_cl], [y_solid], marker="o", ms=6, mfc="#D7263D", mec="white", mew=1.0, zorder=6)
        ax.plot([x_cl, x_cl], [y_solid, y_free], color="#D7263D", linewidth=1.3, linestyle=":", zorder=6)
        if y_free > y_solid + dz:
            ax.annotate(
                "",
                xy=(x_cl, y_free),
                xytext=(x_cl, y_solid),
                arrowprops=dict(arrowstyle="<->", color="#D7263D", lw=1.1),
            )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=6)
    ax.set_xticks(np.round(np.linspace(x0, x1, 4), 2))
    ax.set_yticks(np.round(np.linspace(y0, y1, 4), 2))
    return im


def _plot_profile(ax, phi, chi, p, cl, wall_top):
    """Vertical phi/chi cut through the contact line: the three phases stacked."""
    X, Y = pf.grids(p)
    if cl is None:
        i = p.Nx // 2
        y_solid = wall_top
    else:
        i = int(np.argmin(np.abs(np.asarray(X)[:, 0] - cl[0])))
        y_solid = cl[1]
    y = np.asarray(Y[i])
    lo, hi = max(0.0, y_solid - 0.12), y_solid + 0.55
    ax.axhspan(0.0, 0.5, color="#4a4a4a", alpha=0.22, lw=0)
    ax.axvspan(lo, min(y_solid, hi), color="#4a4a4a", alpha=0.22, lw=0)
    ax.plot(y, np.asarray(phi)[i], color="#08306b", lw=1.8, label=r"$\phi$ (liquid)")
    ax.plot(y, np.asarray(chi)[i], color="#111111", lw=1.2, ls="--", label=r"$\chi$ (solid)")
    ax.axhline(0.5, color="#08306b", lw=0.6, ls=":", alpha=0.8)
    ax.axhline(1.0, color="#111111", lw=0.5, ls=":", alpha=0.5)
    if cl is not None:
        ax.axvline(y_solid, color="#D7263D", lw=1.0, ls=":")
        ax.text(
            y_solid + 0.02,
            0.06,
            "contact\nline",
            color="#D7263D",
            fontsize=6,
            va="bottom",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlabel("height $y$ (near-wall window)", fontsize=7)
    ax.set_ylabel(r"$\phi,\ \chi$", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, linestyle=":")
    if cl is None:
        ax.set_title("free flight (no contact line)", fontsize=6.5)


def build_case(args):
    case = dict(
        surface=args.surface,
        We=200.0,
        Re=200.0,
        cos_theta=0.8,
        R=0.7,
        u_impact=1.7,
        dt=args.dt,
        seed=11,
    )
    if args.surface in ("pillars", "random_pillars"):
        case.update(n_pillars=5, width=0.3, height=0.35)
    return case


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", default="pillars", help="surface family for the close-up")
    ap.add_argument("--N", type=int, default=192, help="solver resolution")
    ap.add_argument("--steps", type=int, default=1200, help="solver steps")
    ap.add_argument("--save-every", type=int, default=4, help="steps between saved frames")
    ap.add_argument("--dt", type=float, default=2.0e-3, help="solver timestep")
    ap.add_argument("--window", type=float, default=0.7, help="close-up window size in length units")
    ap.add_argument("--n-frames", type=int, default=6, help="number of close-up columns")
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    case = build_case(args)
    print(f"[closeup] case: {case}", flush=True)
    p, solid, st = pf.build_case(case, N=args.N, dt=args.dt)
    _, phi, u, v = pf.rollout(st, solid, p, args.steps, save_every=args.save_every)
    phi = np.asarray(phi)
    X, Y = pf.grids(p)
    wall_top = float(np.max(np.where(np.asarray(solid.sdf) < 0.0, Y, -np.inf)))
    print(f"[closeup] wall top y={wall_top:.3f}, frames={phi.shape[0]}, dt={p.dt:g}", flush=True)

    frame_dt = float(p.dt * args.save_every)
    times = np.arange(phi.shape[0]) * frame_dt
    touchdown = touchdown_index(phi, Y)
    if touchdown is None:
        print("[closeup] touchdown: not reached within the simulated horizon", flush=True)
    else:
        print(f"[closeup] touchdown frame={touchdown} (t={times[touchdown]:.3f} s)", flush=True)

    cl_all, wetted, area, angle = [], [], [], []
    for k in range(phi.shape[0]):
        cl = find_contact_line(phi[k], solid.sdf, X, Y)
        cl_all.append(cl)
        wetted.append(wetted_width(phi[k], solid.sdf, Y))
        area.append(float(pf.contact_area(phi[k], solid, p)))
        angle.append(float(pf.measure_contact_angle(phi[k], solid, p)))
    wetted = np.array(wetted)
    area = np.array(area)
    angle = np.array(angle)

    if touchdown is None:
        raise RuntimeError("the drop never descended onto the wall; increase --steps")
    # Airborne frames are never used, whatever the wetting band says.
    contact_idx = np.array(
        [k for k, c in enumerate(cl_all) if c is not None and k >= touchdown],
        dtype=int,
    )
    if len(contact_idx) == 0:
        raise RuntimeError("no wetted contact line detected after touchdown; check --surface/--window")

    # Pick close-up times from touchdown to the end of the wetted phase.  The
    # impact is fast and the capillary retraction is slow -- two orders of
    # magnitude in time -- so the frames are spaced logarithmically.
    lo, hi = int(contact_idx.min()), int(contact_idx.max())
    picks = np.unique(np.rint(np.geomspace(max(lo, touchdown), hi, args.n_frames)).astype(int))
    picks = picks[(picks >= touchdown) & (picks <= hi)]
    print(f"[closeup] close-up frames {picks.tolist()} at t={times[picks].round(3).tolist()}", flush=True)

    dz = float(p.dy)
    fig, axes = plt.subplots(3, len(picks), figsize=(2.9 * len(picks), 7.4), dpi=140)
    axes = np.atleast_2d(axes)
    for c, k in enumerate(picks):
        cl = cl_all[k]
        if cl is not None:
            x_cl, y_solid, _ = cl
            box = (x_cl - args.window / 2, x_cl + args.window / 2, y_solid - dz * 2, y_solid + args.window * 0.85)
        else:
            box = (3.0 - args.window / 2, 3.0 + args.window / 2, wall_top - dz * 2, wall_top + args.window * 0.85)
        box = (max(box[0], 0.0), box[1], max(box[2], 0.0), box[3])

        _plot_full(
            axes[0][c],
            phi[k],
            solid.chi,
            p,
            box,
            f"t = {times[k]:.3f}" if c == 0 else f"t = {times[k]:.3f} s",
            wall_top,
        )
        _plot_closeup(axes[1][c], phi[k], solid.chi, p, box, cl, wall_top, dz)
        _plot_profile(axes[2][c], phi[k], solid.chi, p, cl, wall_top)
        axes[1][c].set_xlabel("x (near-wall zoom)", fontsize=7)

    axes[0][0].set_ylabel("full field", fontsize=8.5, fontweight="bold")
    axes[1][0].set_ylabel("three-phase close-up", fontsize=8.5, fontweight="bold")
    axes[2][0].set_ylabel("cut through contact line", fontsize=8.5, fontweight="bold")

    handles = [
        Patch(facecolor=PHASE_CMAP(c), edgecolor="#999999", linewidth=0.5, label=lab)
        for c, lab in zip((0, 1, 2), PHASE_LABELS)
    ]
    handles.append(Line2D([0], [0], color="#D7263D", marker="o", ms=5, lw=0, label="triple point"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.006),
    )

    fig.suptitle(
        f"Near-wall three-phase contact line on {args.surface} "
        f"(We = 200, Re = 200, $u_0$ = 1.7, $\\cos\\theta$ = 0.8, N = {args.N})\n"
        "liquid (blue) / gas (white) / solid (gray); red marker = triple point where $\\phi$=$\\chi$=0.5",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.93])
    out = os.path.join(FIG_DIR, "fig_contact_closeup.png")
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"[closeup] saved {out}")

    # ------------------------------------------------------------------
    # Time history of the near-wall observables.
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.1), dpi=140)
    axs[0].plot(times, wetted, color="#08306b", lw=1.8)
    axs[0].set_title("wetted width $w_c$", fontsize=9.5, fontweight="bold")
    axs[0].set_ylabel("$w_c$ (length units)", fontsize=8.5)
    axs[1].plot(times, area, color="#2E7D32", lw=1.8)
    axs[1].set_title("near-wall liquid area", fontsize=9.5, fontweight="bold")
    axs[1].set_ylabel(r"$\int_{|sdf|<0.15}\phi\,dA$", fontsize=8.5)
    axs[2].plot(times, angle, color="#D7263D", lw=1.8)
    axs[2].set_title("apparent contact angle", fontsize=9.5, fontweight="bold")
    axs[2].set_ylabel(r"$\theta$ [deg]", fontsize=8.5)
    axs[2].yaxis.get_major_formatter().set_useOffset(False)
    for ax in axs:
        ax.set_xlabel("time $t$ [s]", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        for k in picks:
            ax.axvline(times[k], color="gray", linestyle="--", lw=0.7, alpha=0.5)
    fig.suptitle(
        f"Near-wall wetting observables during impact and spreading ({args.surface}); "
        "dashed lines mark the close-up frames",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out2 = os.path.join(FIG_DIR, "fig_contact_closeup_metrics.png")
    fig.savefig(out2, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"[closeup] saved {out2}")


if __name__ == "__main__":
    main()
