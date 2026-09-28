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
import hashlib
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import surrogate as S


def _source_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _training_code_fingerprint():
    payload = _source_sha256(__file__) + _source_sha256(S.__file__)
    return hashlib.sha256(payload.encode()).hexdigest()


def _accept_unroll_candidate(baseline, candidate, min_improve, max_raw_mass, max_projection_l1):
    """Validation-only guard: a fine-tune must improve rollout without hiding behind projection."""
    if baseline is None or candidate is None:
        return False
    target = baseline["loss"] * (1.0 - min_improve)
    return bool(
        candidate["loss"] < target
        and candidate["raw_mass"] <= max_raw_mass
        and candidate["projection_l1"] <= max_projection_l1
    )


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
    ap.add_argument("--raw-mass-weight", type=float, default=0.02)
    ap.add_argument("--projection-weight", type=float, default=0.01)
    ap.add_argument("--teacher-anchor-weight", type=float, default=0.25)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-improve", type=float, default=0.002)
    ap.add_argument("--guard-raw-mass", type=float, default=0.05)
    ap.add_argument("--guard-projection-l1", type=float, default=0.05)
    ap.add_argument("--disable-unroll-guard", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    code_fingerprint = _training_code_fingerprint()
    params = None
    if args.resume:
        _, params, cfg = S.load_checkpoint(args.resume)
        assert not cfg.get("legacy"), "resume a train_operator.py checkpoint"
        assert cfg.get("dataset_schema_version") == S.DATASET_SCHEMA_VERSION
        if cfg.get("training_code_fingerprint") != code_fingerprint:
            raise RuntimeError(
                "resume checkpoint was produced by different surrogate/training code; "
                "retrain the parent checkpoint before unrolled fine-tuning"
            )
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
    if args.resume and cfg.get("train_dataset_fingerprint") != store.dataset_fingerprint:
        raise RuntimeError(
            "resume checkpoint/data mismatch: "
            f"checkpoint={cfg.get('train_dataset_fingerprint')!r}, current={store.dataset_fingerprint!r}"
        )

    uv = cfg.get("uv_scale") or store.uv_scale
    cfg["uv_scale"] = uv
    cfg["data"] = args.data
    cfg["n_train_traj"] = len(store)
    cfg["train_dataset_fingerprint"] = store.dataset_fingerprint
    cfg["training_code_fingerprint"] = code_fingerprint
    cfg["parent_checkpoint"] = args.resume
    cfg["train_unroll"] = int(args.unroll)
    cfg["train_lr"] = float(args.lr)
    cfg["raw_mass_weight"] = float(args.raw_mass_weight)
    cfg["projection_weight"] = float(args.projection_weight)
    cfg["teacher_anchor_weight"] = float(args.teacher_anchor_weight)

    K = args.unroll
    all_win = store.windows(K)
    if len(store) < 2 or not 0.0 < args.val_frac < 0.5:
        raise ValueError("--val-frac must be in (0, 0.5) and training requires at least two trajectories")
    split_rng = np.random.default_rng(args.seed + 9173)
    case_ids = np.arange(len(store))
    split_rng.shuffle(case_ids)
    n_val = min(len(store) - 1, max(1, int(round(len(store) * args.val_frac))))
    val_cases = np.sort(case_ids[:n_val])
    val_mask = np.isin(all_win[:, 0], val_cases)
    val_win = all_win[val_mask]
    win = all_win[~val_mask]
    if not len(win) or not len(val_win):
        raise RuntimeError("deterministic train/validation split produced an empty window set")
    cfg["validation_case_indices"] = [int(x) for x in val_cases]
    cfg["n_source_traj"] = len(store)
    cfg["n_validation_traj"] = int(n_val)
    cfg["n_train_traj"] = int(len(store) - n_val)
    cfg["n_train_windows"] = int(len(win))
    cfg["n_validation_windows"] = int(len(val_win))

    fam_counts = {f: sum(m["family"] == f for m in store.meta) for f in ("simple", "complex")}
    print(
        f"trajectories={len(store)} {fam_counts} T={store.T} K={K} "
        f"train_windows={len(win)} val_windows={len(val_win)} uv={uv:.3f}",
        flush=True,
    )

    model = S.build_model(cfg)
    G = store.geom.shape[-1]
    H, W = store.H, store.W
    if params is None:
        params = model.init(jax.random.PRNGKey(args.seed), jnp.ones((1, H, W, 3 + G)), jnp.ones((1, 4)))["params"]
    npar = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"model={cfg['arch']} geom={cfg['geom']} params={npar}", flush=True)

    sched = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, min(100, max(1, args.steps // 10)), args.steps, args.lr * 0.05
    )
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adamw(sched, weight_decay=args.wd))
    opt_state = tx.init(params)
    wts = jnp.array([2.0, 1.0, 1.0])
    anchor_weight = args.teacher_anchor_weight if K > 1 else 0.0

    def objective(p, fr, geom, cond, key, noise_std):
        state = fr[:, 0] + noise_std * jax.random.normal(key, fr[:, 0].shape)
        total = 0.0
        metric_sum = jnp.zeros((4,), dtype=fr.dtype)
        for k in range(1, fr.shape[1]):
            x = jnp.concatenate([state, geom], axis=-1)
            pred, aux = model.apply({"params": p}, x, cond, return_aux=True)
            state_loss = jnp.mean(wts * (pred - fr[:, k]) ** 2)
            raw_mass = jnp.mean(aux["raw_mass_rel"])
            projection_l1 = jnp.mean(aux["projection_l1_rel"])

            anchor_loss = jnp.asarray(0.0, dtype=fr.dtype)
            if anchor_weight > 0.0:
                x_true = jnp.concatenate([fr[:, k - 1], geom], axis=-1)
                teacher = model.apply({"params": p}, x_true, cond)
                anchor_loss = jnp.mean(wts * (teacher - fr[:, k]) ** 2)

            total = total + state_loss + anchor_weight * anchor_loss
            total = total + args.raw_mass_weight * raw_mass
            total = total + args.projection_weight * projection_l1
            metric_sum = metric_sum + jnp.array([state_loss, anchor_loss, raw_mass, projection_l1], dtype=fr.dtype)
            state = pred

        denom = fr.shape[1] - 1
        return total / denom, metric_sum / denom

    @jax.jit
    def train_step(p, o, fr, geom, cond, key):
        (lv, metrics), g = jax.value_and_grad(objective, has_aux=True)(p, fr, geom, cond, key, args.noise)
        upd, o = tx.update(g, o, p)
        return optax.apply_updates(p, upd), o, lv, metrics

    @jax.jit
    def eval_step(p, fr, geom, cond):
        return objective(p, fr, geom, cond, jax.random.PRNGKey(0), 0.0)

    def validation_metrics(p):
        max_windows = max(args.batch, args.val_batches * args.batch)
        if len(val_win) > max_windows:
            ids = np.linspace(0, len(val_win) - 1, max_windows, dtype=np.int64)
            chosen = val_win[ids]
        else:
            chosen = val_win
        accum = np.zeros(5, dtype=np.float64)
        count = 0
        for start in range(0, len(chosen), args.batch):
            b = chosen[start : start + args.batch]
            fr, geom, cond = store.batch(b, K, uv)
            lv, metrics = eval_step(p, jnp.asarray(fr), jnp.asarray(geom), jnp.asarray(cond))
            n = len(b)
            accum[0] += float(lv) * n
            accum[1:] += np.asarray(metrics, dtype=np.float64) * n
            count += n
        vals = accum / max(count, 1)
        return {
            "loss": float(vals[0]),
            "state": float(vals[1]),
            "anchor": float(vals[2]),
            "raw_mass": float(vals[3]),
            "projection_l1": float(vals[4]),
        }

    def copy_params(p):
        return jax.tree_util.tree_map(lambda x: x.copy(), p)

    def save_checkpoint(path, p, cfg_obj):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(dict(params=p, cfg=cfg_obj, uv_scale=uv), fh)

    guard_active = bool(args.resume and K > 1 and not args.disable_unroll_guard)
    parent_params = copy_params(params) if guard_active else None
    baseline_val = validation_metrics(params) if guard_active else None
    if baseline_val is not None:
        print(f"unroll baseline validation: {baseline_val}", flush=True)

    best_candidate = None
    best_candidate_params = None
    bad_checks = 0
    t0, run, run_metrics, hist = time.time(), 0.0, np.zeros(4), []
    key = jax.random.PRNGKey(args.seed + 1)

    for it in range(1, args.steps + 1):
        b = win[rng.integers(0, len(win), size=args.batch)]
        fr, geom, cond = store.batch(b, K, uv)
        key, sk = jax.random.split(key)
        params, opt_state, lv, metrics = train_step(
            params, opt_state, jnp.asarray(fr), jnp.asarray(geom), jnp.asarray(cond), sk
        )
        run += float(lv)
        run_metrics += np.asarray(metrics, dtype=np.float64)

        if it % args.log_every == 0:
            mean_metrics = run_metrics / args.log_every
            hist.append(
                {
                    "step": it,
                    "loss": run / args.log_every,
                    "state": float(mean_metrics[0]),
                    "anchor": float(mean_metrics[1]),
                    "raw_mass": float(mean_metrics[2]),
                    "projection_l1": float(mean_metrics[3]),
                }
            )
            el = time.time() - t0
            print(
                f"step {it:5d} loss={run / args.log_every:.3e} "
                f"state={mean_metrics[0]:.3e} rawM={mean_metrics[2]:.3e} "
                f"projL1={mean_metrics[3]:.3e} "
                f"({el:.0f}s, eta {el / it * (args.steps - it):.0f}s)",
                flush=True,
            )
            run = 0.0
            run_metrics[:] = 0.0

        if it % args.val_every == 0 or it == args.steps:
            current_val = validation_metrics(params)
            print(f"validation step={it}: {current_val}", flush=True)
            if best_candidate is None or current_val["loss"] < best_candidate["loss"]:
                best_candidate = current_val
                best_candidate_params = copy_params(params)
                bad_checks = 0
            else:
                bad_checks += 1
            if bad_checks >= args.patience:
                print(f"early stop after {bad_checks} non-improving validation checks", flush=True)
                break

    if best_candidate_params is None:
        best_candidate_params = copy_params(params)
        best_candidate = validation_metrics(params)

    accepted = True
    final_params = best_candidate_params
    if guard_active:
        accepted = _accept_unroll_candidate(
            baseline_val,
            best_candidate,
            args.min_improve,
            args.guard_raw_mass,
            args.guard_projection_l1,
        )
        cfg["unroll_accepted"] = bool(accepted)
        cfg["validation_baseline"] = baseline_val
        cfg["validation_best_candidate"] = best_candidate
        if accepted:
            cfg["effective_model"] = "unroll_candidate"
        else:
            cfg["effective_model"] = "parent_checkpoint"
            rejected_path = os.path.splitext(args.out)[0] + ".rejected.pkl"
            rejected_cfg = dict(cfg)
            rejected_cfg["effective_model"] = "rejected_unroll_candidate"
            save_checkpoint(rejected_path, best_candidate_params, rejected_cfg)
            final_params = parent_params
            print(
                f"unroll candidate REJECTED by validation guard; saved diagnostic candidate to {rejected_path}",
                flush=True,
            )
    else:
        cfg["validation_best_candidate"] = best_candidate

    cfg["train_hist"] = hist
    save_checkpoint(args.out, final_params, cfg)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump(cfg, fh, indent=1)
    print(
        f"saved {args.out} accepted={cfg.get('unroll_accepted', True)} ({time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
