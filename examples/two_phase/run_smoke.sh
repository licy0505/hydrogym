#!/usr/bin/env bash
# Deterministic CPU L0 contract smoke: data -> tiny FNO -> evaluation.
# This is a pipeline/regression test, not a CFD accuracy benchmark.
set -euo pipefail

cd "$(dirname "$0")"
PY=${PY:-python}
ARTIFACTS=${SMOKE_ARTIFACTS:-artifacts/smoke}
DATA="$ARTIFACTS/data"
CKPT="$ARTIFACTS/checkpoints/fno_smoke.pkl"
REPORT="$ARTIFACTS/report.json"

export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

# Always start from an isolated artifact tree.  In particular, this prevents a
# stale trajectory or checkpoint from making a partial smoke run look healthy.
rm -rf "$ARTIFACTS"
mkdir -p "$DATA" "$(dirname "$CKPT")"

$PY generate_dataset.py \
  --set smoke \
  --out "$DATA" \
  --N 64 \
  --ds 1 \
  --nsteps 220 \
  --save_every 10 \
  --dt 2e-3 \
  --min-feature-cells 3 \
  --require-complete

$PY - "$DATA/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as fh:
    manifest = json.load(fh)
expected = {"expected": 8, "accepted": 8, "rejected": 0, "complete": True}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"manifest {key}={manifest.get(key)!r}, expected {value!r}")
print(
    f"manifest expected={manifest['expected']} accepted={manifest['accepted']} "
    f"rejected={manifest['rejected']} complete={str(manifest['complete']).lower()}"
)
PY

$PY - "$DATA" <<'PY'
import glob
import json
import sys

import numpy as np

files = glob.glob(sys.argv[1] + "/train_smoke_*_flat.npz")
if not files:
    raise SystemExit("no smoke flat training case found")
with np.load(sorted(files)[0], allow_pickle=True) as d:
    phi = np.asarray(d["phi"], dtype=np.float64)
    metadata = json.loads(str(np.asarray(d["case"]).item()))
# The first saved frame is after ten solver steps.  Compare it to the last
# frame and use physical cell-centre coordinates for the vertical centroid.
dx = float(metadata["saved_dx"])
y = (np.arange(phi.shape[2]) + 0.5) * dx
mass = np.maximum(phi.sum(axis=(1, 2)), 1e-12)
centroid_y = (phi * y[None, None, :]).sum(axis=(1, 2)) / mass
if not centroid_y[-1] < centroid_y[0]:
    raise SystemExit(f"flat liquid centroid did not move downward: {centroid_y[0]} -> {centroid_y[-1]}")
near_wall = (y > 0.25) & (y < 0.75)
near_wall_signal = float(np.max(phi[-1, :, near_wall]))
if not near_wall_signal > 1e-4:
    raise SystemExit(f"no nonzero near-wall liquid signal: {near_wall_signal}")
print(f"kinematic centroid_y={centroid_y[0]:.6g}->{centroid_y[-1]:.6g} near_wall={near_wall_signal:.6g}")
PY

$PY train_operator.py \
  --data "$DATA" \
  --families simple \
  --profile smoke \
  --arch fno \
  --geom sdf \
  --width 8 \
  --modes 4 \
  --layers 2 \
  --steps 20 \
  --batch 2 \
  --log-every 10 \
  --val-every 10 \
  --val-batches 1 \
  --patience 4 \
  --out "$CKPT"

if [[ "${SMOKE_UNROLL:-0}" == "1" ]]; then
  $PY train_operator.py \
    --data "$DATA" \
    --families simple \
    --profile smoke \
    --resume "$CKPT" \
    --unroll 2 \
    --steps 3 \
    --batch 2 \
    --log-every 3 \
    --val-every 3 \
    --val-batches 1 \
    --patience 1 \
    --out "$ARTIFACTS/checkpoints/fno_smoke_unroll.pkl"
fi

$PY - "$DATA" "$CKPT" "$REPORT" <<'PY'
import json
import sys

import numpy as np

import evaluate_transfer as E


data, checkpoint, report_path = sys.argv[1:]
persistence = E.evaluate("persistence", data, horizon=5, verbose=False)
model = E.evaluate(checkpoint, data, horizon=5, verbose=False)

def mean_metric(result, key):
    values = [float(row[key]) for row in result["rows"]]
    if not values or not np.isfinite(values).all():
        raise SystemExit(f"non-finite or missing metric {key}")
    return float(np.mean(values))

persistence_error = mean_metric(persistence, "rollK")
model_error = mean_metric(model, "rollK")
for row in model["rows"]:
    if not np.isfinite(row["iouK"]) or not 0.0 <= row["iouK"] <= 1.0:
        raise SystemExit(f"IoU out of range: {row}")
skill = 1.0 - model_error / max(persistence_error, 1e-12)
report = {
    "status": "PASS",
    "profile": "smoke",
    "publication_quality": False,
    "horizon_saved_frames": 5,
    "persistence_rollout_error": persistence_error,
    "model_rollout_error": model_error,
    "persistence_relative_skill": float(skill),
    "skill": float(skill),
    "metrics_finite": True,
    "iou_in_range": True,
}
with open(report_path, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"report status={report['status']}")
print(json.dumps(report, indent=2))
PY
