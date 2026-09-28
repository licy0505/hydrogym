"""
Collect ``results/*.json`` (from evaluate_transfer.py) into a comparison table and
figures:

* ``results/summary.md``          -- markdown table of all evaluated models
* ``figures/fig_fno_compare.png``  -- rollout RMSE / IoU, simple (train) vs complex (test)
* ``figures/fig_fno_rollout.png``  -- qualitative rollouts on unseen complex surfaces

Usage::

    python compare_models.py [--models persistence,unet_legacy_base,fno_sdf_large_u3,...]
                             [--show unet_legacy_base,fno_sdf_aug_u3]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import evaluate_transfer as E
import matplotlib.pyplot as plt
import surrogate as S

LABELS = {
    "persistence": "persistence (φ(t+1)=φ(t))",
    "unet_legacy_base": "U-Net·χ, base-13 (old)",
    "unet_legacy_base_u3": "U-Net·χ, base-13 + unroll",
    "fno_chi_large": "FNO·χ, large",
    "fno_chi_large_u3": "FNO·χ, large + unroll",
    "fno_sdf_large": "FNO·SDF, large",
    "fno_sdf_large_u3": "FNO·SDF, large + unroll",
    "fno_sdf_aug": "FNO·SDF, large+aug",
    "fno_sdf_aug_u3": "FNO·SDF, large+aug + unroll",
    "lowWe_persistence": "low-We persistence",
    "lowWe_fno_sdf": "low-We FNO·SDF",
    "lowWe_fno_sdf_u3": "low-We FNO·SDF + guarded unroll",
}


def flip(a):
    return np.flipud(np.asarray(a).T)


def load_results(names):
    out = {}
    for n in names:
        f = f"results/{n}.json"
        if os.path.exists(f):
            with open(f) as fh:
                out[n] = json.load(fh)
    return out


def table(res):
    first = next(iter(res.values()))
    K = first["horizon"]
    full = first.get("full_horizon", first["T"] - 1)
    hdr = (
        f"| model | train traj | group | 1-step RMSE | rollout-{K} RMSE | rollout-{full} RMSE "
        f"| IoU@{K} | raw mass err | projection L1 |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for n, r in res.items():
        for g in ("train/simple", "test/complex"):
            s = r["summary"].get(g)
            if s is None:
                continue
            lines.append(
                f"| {LABELS.get(n, n)} | {r['cfg'].get('n_train_traj', 13 if n != 'persistence' else '-')} | {g} | "
                f"{s['one_step']:.4f} | {s['rollK']:.4f} | {s['rollT']:.4f} | {s['iouK']:.3f} "
                f"| {s.get('rawMassK', float('nan')):.2e} | {s.get('projL1K', float('nan')):.2e} |"
            )
    return hdr + "\n".join(lines) + "\n"


def fig_compare(res, path):
    names = list(res)
    groups = ("train/simple", "test/complex")
    metrics = (("rollK", "rollout-40 φ-RMSE ↓"), ("iouK", "interface IoU @40 ↑"), ("massK", "liquid-mass error ↓"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.2))
    x = np.arange(len(names))
    for ax, (m, title) in zip(axes, metrics):
        for j, g in enumerate(groups):
            vals = [res[n]["summary"].get(g, {}).get(m, np.nan) for n in names]
            ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, label=g.replace("/", " · "), color=["#4c72b0", "#dd8452"][j])
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(n, n) for n in names], rotation=35, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_rollout(show, path, frames=(10, 25, 40, 70), data="data/base"):
    cs = S.load_full(data, "test")
    picks, seen = [], set()
    for c in cs:  # one case per complex family
        if c["surface"] not in seen:
            seen.add(c["surface"])
            picks.append(c)
    preds = {}
    for name in show:
        predict, cfg, gm = E.make_predictor(f"ckpts/{name}.pkl")
        geom = np.stack([S.geometry_features(c["chi"], c["sdf"], float(c["scalars"][3]), gm) for c in picks])
        scal = np.stack([c["scalars"] for c in picks])
        st = np.stack([np.stack([c["phi"][0], c["u"][0], c["v"][0]], -1) for c in picks])
        traj = [st[..., 0]]
        for _ in range(max(frames)):
            st = predict(st, geom, scal)
            traj.append(st[..., 0])
        preds[name] = np.stack(traj, 1)
    nr, nc = len(picks) * len(show), len(frames)
    fig, axes = plt.subplots(nr, nc, figsize=(2.0 * nc + 1.2, 2.0 * nr))
    r = 0
    for i, c in enumerate(picks):
        for name in show:
            for j, t in enumerate(frames):
                ax = axes[r][j]
                ax.imshow(flip(c["chi"]), cmap="gray_r", vmin=0, vmax=1)
                p = preds[name][i, t]
                ax.imshow(np.ma.masked_where(flip(p) < 0.5, flip(p)), cmap="Blues", vmin=0, vmax=1, alpha=0.9)
                ax.contour(flip(c["phi"][t]), levels=[0.5], colors="r", linewidths=0.8)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"frame {t}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{c['surface']}\n{LABELS.get(name, name)}", fontsize=6)
            r += 1
    fig.suptitle("unseen complex surfaces — surrogate rollout (blue) vs solver truth (red contour)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--show", default="unet_legacy_base,fno_sdf_aug_u3")
    ap.add_argument("--tag", default="", help="suffix used for summary/figure outputs")
    ap.add_argument("--data", default="", help="override rollout dataset for qualitative figure")
    args = ap.parse_args()
    if args.models:
        names = args.models.split(",")
    else:
        order = list(LABELS)
        found = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob("results/*.json")]
        names = [n for n in order if n in found] + sorted(n for n in found if n not in order)
    res = load_results(names)
    if not res:
        raise RuntimeError("no requested result JSON files were found")

    fingerprints = {r.get("protocol", {}).get("eval_dataset_fingerprint", "legacy") for r in res.values()}
    if len(fingerprints) != 1:
        raise RuntimeError(
            "refusing to compare mixed evaluation protocols/datasets: "
            f"{sorted(fingerprints)}. Pass --models with one protocol at a time."
        )

    data_paths = {r.get("protocol", {}).get("data", "") for r in res.values()}
    data_paths.discard("")
    data = args.data or (next(iter(data_paths)) if len(data_paths) == 1 else "")
    if not data and args.show:
        raise RuntimeError("qualitative comparison needs --data for legacy/ambiguous result metadata")

    suffix = f"_{args.tag}" if args.tag else ""
    md = table(res)
    os.makedirs("results", exist_ok=True)
    summary_path = f"results/summary{suffix}.md"
    with open(summary_path, "w") as fh:
        fh.write(md)
    print(md)

    os.makedirs("figures", exist_ok=True)
    compare_path = f"figures/fig_fno_compare{suffix}.png"
    fig_compare(res, compare_path)
    show = [s for s in args.show.split(",") if s in res and os.path.exists(f"ckpts/{s}.pkl")]
    rollout_path = f"figures/fig_fno_rollout{suffix}.png"
    if show:
        fig_rollout(show, rollout_path, data=data)
    print(f"wrote {summary_path}, {compare_path}" + (f", {rollout_path}" if show else ""))


if __name__ == "__main__":
    main()
