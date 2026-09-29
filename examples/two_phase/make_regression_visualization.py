"""
FNO autoregressive regression evaluation on the four *unseen* spreading walls.

The ``spreading`` case set (:mod:`cases`) trains the surrogate on flat walls and
regular pillar arrays only, and reserves four complex geometries for testing:

    random_pillars | hierarchical | grooves | wedge

This script runs the trained FNO **autoregressively** over the full horizon of
each test trajectory -- one forward pass at a time, feeding the model's own
output back as the next input -- and compares it against the phase-field solver
truth field by field, frame by frame.

Three figures are written to ``figures/``:

* ``fig_fno_regression.png``          4 unseen walls x snapshots, FNO (blue
  contour / fill) over solver truth (red contour), with the per-frame RMSE.
* ``fig_fno_regression_metrics.png``  spreading width D(t), fluid mass M/M0 and
  per-surface curves.
* ``fig_fno_regression_summary.png``  per-surface error bar summary.

Nothing here is test-set-tuned: the checkpoint is whatever ``--ckpt`` points at
and the trajectories are read straight from ``--data``.

Usage::

    python generate_dataset.py --set spreading --out data/spreading --nsteps 2000 --ds 3
    python train_operator.py --data data/spreading --arch fno --geom sdf \\
        --steps 2500 --out ckpts/fno_spreading.pkl
    python train_operator.py --data data/spreading --resume ckpts/fno_spreading.pkl \\
        --unroll 3 --steps 600 --out ckpts/fno_spreading_u3.pkl
    python make_regression_visualization.py --data data/spreading \\
        --ckpt ckpts/fno_spreading_u3.pkl

``--ckpt`` defaults to the guarded unrolled model, which is the published model
whenever ``train_operator.py`` accepted the candidate (same convention as
``run_experiments.sh``).  Pass ``--ckpt ckpts/fno_spreading.pkl`` for the
teacher-forced parent.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import surrogate as S
from evaluate_transfer import iou, make_predictor
from matplotlib.lines import Line2D
from visualize import OSDIR, overlay, spreading

UNSEEN = ("random_pillars", "hierarchical", "grooves", "wedge")


def pick_cases(data_dir):
    """One test trajectory per unseen surface family, in a stable order."""
    tests = S.load_full(data_dir, "test")
    picked = []
    for fam in UNSEEN:
        matches = [c for c in tests if c["surface"] == fam]
        if matches:
            picked.append(matches[0])
    missing = [f for f in UNSEEN if not any(c["surface"] == f for c in picked)]
    if missing:
        raise RuntimeError(f"no TEST trajectories for {missing} in {data_dir!r}")
    return picked


def rollout(predict, geom_mode, c):
    """Autoregressive full-horizon rollout starting from the true first frame."""
    geom = S.geometry_features(c["chi"], c["sdf"], float(c["scalars"][3]), geom_mode)
    st = np.stack([c["phi"][0], c["u"][0], c["v"][0]], -1)
    traj = [st[..., 0].copy()]
    for _ in range(c["phi"].shape[0] - 1):
        st = predict(st[None], geom[None], c["scalars"][None])[0]
        traj.append(st[..., 0].copy())
    return np.stack(traj)


def _frames(T, n_snap):
    """Snapshot frame indices, geometrically spaced.

    Impact and spreading are finished long before the horizon ends, so a uniform
    spacing spends most panels on the near-static tail.  ``np.geomspace`` cannot
    start at zero, hence the explicit log ramp.
    """
    if T <= 1:
        return np.array([0], dtype=int)
    ramp = np.logspace(0.0, 1.0, max(2, n_snap)) - 1.0
    return np.unique(np.rint((T - 1) * ramp / ramp[-1]).astype(int))


def fig_regression(rollouts, n_snap=6):
    n_rows = len(rollouts)
    fig, axes = plt.subplots(n_rows, n_snap, figsize=(2.55 * n_snap, 2.35 * n_rows), dpi=140)
    axes = np.atleast_2d(axes)

    for r, item in enumerate(rollouts):
        c, pred = item["case"], item["pred"]
        truth = c["phi"]
        T = min(len(pred), truth.shape[0])
        for j, ti in enumerate(_frames(T, n_snap)):
            ax = axes[r][j]
            overlay(ax, pred[ti], c["chi"], truth=truth[ti], display_thresh=0.05, truth_levels=(0.5,))
            rmse = float(np.sqrt(np.mean((pred[ti] - truth[ti]) ** 2)))
            ax.set_title(f"t = {float(c['time'][ti]):.3f} s\nRMSE = {rmse:.4f}", fontsize=7.5)
            if j == 0:
                ax.text(
                    0.03,
                    0.03,
                    "FNO",
                    transform=ax.transAxes,
                    fontsize=6.5,
                    color="#0D47A1",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#0D47A1", alpha=0.9),
                )
                ax.text(
                    0.97,
                    0.03,
                    "truth",
                    transform=ax.transAxes,
                    fontsize=6.5,
                    color="red",
                    ha="right",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="red", alpha=0.9),
                )
                ax.set_ylabel(f"{c['surface']}\nIoU = {item['iou']:.3f}", fontsize=8.5, fontweight="bold")

    handles = [
        Line2D([0], [0], color="#0D47A1", lw=2.0, label="FNO rollout"),
        Line2D([0], [0], color="red", lw=2.0, label="solver truth"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.002))

    fig.suptitle(
        "FNO full-horizon autoregressive rollout (blue) vs phase-field solver truth (red)\n"
        "trained on flat walls + regular pillar arrays; all four surfaces below are unseen",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    out = os.path.join(OSDIR, "fig_fno_regression.png")
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def fig_metrics(rollouts):
    n = len(rollouts)
    fig = plt.figure(figsize=(4.4 * n, 8.0), dpi=140)
    gs = fig.add_gridspec(2, n, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.32)

    for i, item in enumerate(rollouts):
        c, pred = item["case"], item["pred"]
        T = min(len(pred), c["phi"].shape[0])
        tt = np.asarray(c["time"][:T])
        fluid = (c["sdf"] >= 0.0).astype(np.float32)
        m0 = max(float(np.sum(c["phi"][0] * fluid)), 1e-12)
        d_true = np.array([spreading(c["phi"][t]) for t in range(T)])
        d_pred = np.array([spreading(pred[t]) for t in range(T)])
        m_true = np.array([float(np.sum(c["phi"][t] * fluid) / m0) for t in range(T)])
        m_pred = np.array([float(np.sum(pred[t] * fluid) / m0) for t in range(T)])

        ax = fig.add_subplot(gs[0, i])
        ax.plot(tt, d_true, "r-", lw=2.0, label="truth")
        ax.plot(tt, d_pred, color="#0D47A1", lw=2.0, ls="--", label="FNO")
        ax.set_title(f"$D(t)$ — {c['surface']}", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("time [s]", fontsize=8.5)
        ax.set_ylabel("spreading width", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[1, i])
        ax.plot(tt, m_true, "r-", lw=2.0)
        ax.plot(tt, m_pred, color="#0D47A1", lw=2.0, ls="--")
        ax.axhline(1.0, color="k", ls=":", lw=0.9)
        ax.set_title(f"fluid mass $M/M_0$ — {c['surface']}", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("time [s]", fontsize=8.5)
        ax.set_ylabel("$M/M_0$", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(
        "FNO spreading diagnostics on unseen walls: footprint $D(t)$ and fluid mass $M/M_0$",
        fontsize=11.5,
        fontweight="bold",
    )
    out = os.path.join(OSDIR, "fig_fno_regression_metrics.png")
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def fig_summary(rollouts):
    """Compact per-surface error bar chart, one panel per metric (very different scales)."""
    names = [it["case"]["surface"] for it in rollouts]
    x = np.arange(len(names))
    series = (
        ("full-horizon rollout RMSE", [it["roll_rmse"] for it in rollouts], "#0D47A1"),
        ("final IoU", [it["iou"] for it in rollouts], "#2E7D32"),
        ("fluid mass error (rel.)", [it["mass_err"] for it in rollouts], "#D7263D"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), dpi=140)
    for ax, (label, vals, color) in zip(np.atleast_1d(axes), series):
        ax.bar(x, vals, width=0.55, color=color)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.4g}", ha="center", va="bottom", fontsize=7.5)
        ax.set_title(label, fontsize=9.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8, rotation=18, ha="right")
        ax.set_ylim(0.0, max(vals) * 1.22)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.suptitle(
        "FNO regression summary over the full horizon, four unseen walls",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = os.path.join(OSDIR, "fig_fno_regression_summary.png")
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/spreading", help="dataset dir with train + test splits")
    ap.add_argument("--ckpt", default="ckpts/fno_spreading_u3.pkl", help="trained FNO checkpoint")
    ap.add_argument("--frames", type=int, default=6, help="snapshots per surface in the field figure")
    args = ap.parse_args()

    os.makedirs(OSDIR, exist_ok=True)
    predict, cfg, geom_mode = make_predictor(args.ckpt)
    print(f"loaded {args.ckpt}: arch={cfg.get('arch')} geom={cfg.get('geom')} trained_on={cfg.get('data')}")

    cases = pick_cases(args.data)
    rollouts = []
    for c in cases:
        pred = rollout(predict, geom_mode, c)
        T = min(len(pred), c["phi"].shape[0])
        pred, truth = pred[:T], c["phi"][:T]
        fluid = (c["sdf"] >= 0.0).astype(np.float32)
        m0 = max(float(np.sum(truth[0] * fluid)), 1e-12)
        item = dict(
            case=c,
            pred=pred,
            roll_rmse=float(np.sqrt(np.mean((pred - truth) ** 2))),
            iou=iou(pred[-1], truth[-1]),
            mass_err=float(
                np.mean([abs(np.sum(pred[t] * fluid) - np.sum(truth[t] * fluid)) / m0 for t in range(1, T)])
            ),
        )
        rollouts.append(item)
        print(
            f"  {c['surface']:15s} rollout RMSE={item['roll_rmse']:.4f} "
            f"final IoU={item['iou']:.3f} mass err={item['mass_err']:.4f}",
            flush=True,
        )

    fig_regression(rollouts, n_snap=args.frames)
    fig_metrics(rollouts)
    fig_summary(rollouts)


if __name__ == "__main__":
    main()
