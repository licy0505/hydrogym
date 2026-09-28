#!/usr/bin/env bash
# End-to-end FNO / SDF / fingerprinted experiment chain with guarded unroll fine-tuning.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
STEPS=${STEPS:-3000}
USTEPS=${USTEPS:-600}
mkdir -p logs ckpts results

for s in base large aug; do
  $PY generate_dataset.py --set "$s" --out "data/$s" --nsteps 2000 --ds 3 \
    --min-feature-cells 2.0 | tee -a "logs/gen_$s.log"
done
$PY generate_dataset.py --set lowWe_all --out data/lowWe_all --nsteps 2000 --ds 2 \
  --min-feature-cells 2.5 | tee -a logs/gen_lowWe_all.log

ckpt_current() {
  local f=$1 data=$2 fams=${3:-} expected_unroll=$4
  [ -f "$f" ] || return 1
  $PY - "$f" "$data" "$fams" "$expected_unroll" <<'PYIN'
import glob
import hashlib
import os
import pickle
import sys

import numpy as np

path, data, fams_s, expected_unroll = sys.argv[1:5]
families = {x for x in fams_s.split(",") if x}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


try:
    with open(path, "rb") as fh:
        ck = pickle.load(fh)
    cfg = ck.get("cfg") or {}
    if int(cfg.get("dataset_schema_version", 0)) != 3:
        raise ValueError("schema")

    fps = []
    for d in [x for x in data.split(",") if x]:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            with np.load(f, allow_pickle=True) as z:
                if str(z["split"]) != "train":
                    continue
                fam = "simple" if str(z["surface"]) in ("flat", "pillars") else "complex"
                if families and fam not in families:
                    continue
                if int(np.asarray(z["dataset_schema_version"]).item()) != 3:
                    raise ValueError("data schema")
                fps.append(str(np.asarray(z["dataset_fingerprint"]).item()))
    current_data = hashlib.sha256("\n".join(sorted(fps)).encode()).hexdigest()
    code_payload = sha256("train_operator.py") + sha256("surrogate.py")
    current_code = hashlib.sha256(code_payload.encode()).hexdigest()
    ok = (
        bool(fps)
        and cfg.get("train_dataset_fingerprint") == current_data
        and cfg.get("training_code_fingerprint") == current_code
        and int(cfg.get("train_unroll", 0)) == int(expected_unroll)
    )
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PYIN
}

unroll_accepted() {
  local f=$1
  [ -f "$f" ] || return 1
  $PY - "$f" <<'PYIN'
import pickle
import sys
try:
    with open(sys.argv[1], "rb") as fh:
        cfg = (pickle.load(fh).get("cfg") or {})
    ok = cfg.get("unroll_accepted") is True
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PYIN
}

train() {
  local name=$1 data=$2 eval_data=$3 fams=$4; shift 4
  local fam_args=()
  [ -n "$fams" ] && fam_args=(--families "$fams")

  if ! ckpt_current "ckpts/$name.pkl" "$data" "$fams" 1; then
    rm -f "ckpts/$name.pkl" "ckpts/${name}_u3.pkl" "ckpts/${name}_u3.rejected.pkl"
    rm -f "results/$name.json" "results/${name}_u3.json"
    $PY train_operator.py --data "$data" "${fam_args[@]}" --steps "$STEPS" \
      --out "ckpts/$name.pkl" "$@" 2>&1 | tee "logs/train_$name.log"
  fi

  if ! ckpt_current "ckpts/${name}_u3.pkl" "$data" "$fams" 3; then
    rm -f "results/${name}_u3.json" "ckpts/${name}_u3.rejected.pkl"
    $PY train_operator.py --data "$data" "${fam_args[@]}" --resume "ckpts/$name.pkl" \
      --unroll 3 --batch 8 --lr 1e-4 --steps "$USTEPS" \
      --raw-mass-weight 0.05 --projection-weight 0.02 --teacher-anchor-weight 0.25 \
      --val-frac 0.10 --min-improve 0.002 --guard-raw-mass 0.05 \
      --guard-projection-l1 0.05 --out "ckpts/${name}_u3.pkl" \
      2>&1 | tee "logs/train_${name}_u3.log"
  fi

  $PY evaluate_transfer.py --data "$eval_data" --ckpt "ckpts/$name.pkl" \
    --json "results/$name.json" | tee "logs/eval_$name.log"

  if unroll_accepted "ckpts/${name}_u3.pkl"; then
    $PY evaluate_transfer.py --data "$eval_data" --ckpt "ckpts/${name}_u3.pkl" \
      --json "results/${name}_u3.json" | tee "logs/eval_${name}_u3.log"
  else
    rm -f "results/${name}_u3.json"
    echo "[$name] guarded unroll rejected; parent checkpoint remains the published model" \
      | tee "logs/eval_${name}_u3.log"
  fi
}

$PY evaluate_transfer.py --data data/base --ckpt persistence --json results/persistence.json
$PY evaluate_transfer.py --data data/lowWe_all --ckpt persistence \
  --json results/lowWe_persistence.json

train fno_sdf_large data/base,data/large data/base "" --arch fno --geom sdf
train fno_chi_large data/base,data/large data/base "" --arch fno --geom chi
train fno_sdf_aug data/base,data/large,data/aug data/base "" --arch fno --geom sdf
train lowWe_fno_sdf data/lowWe_all data/lowWe_all simple --arch fno --geom sdf

$PY compare_models.py \
  --models persistence,fno_sdf_large,fno_sdf_large_u3,fno_chi_large,fno_chi_large_u3,fno_sdf_aug,fno_sdf_aug_u3 \
  --show fno_sdf_large_u3,fno_sdf_aug_u3 --tag base --data data/base
$PY compare_models.py \
  --models lowWe_persistence,lowWe_fno_sdf,lowWe_fno_sdf_u3 \
  --show lowWe_fno_sdf,lowWe_fno_sdf_u3 --tag lowWe --data data/lowWe_all

LOWWE_CKPT=ckpts/lowWe_fno_sdf.pkl
if unroll_accepted ckpts/lowWe_fno_sdf_u3.pkl; then
  LOWWE_CKPT=ckpts/lowWe_fno_sdf_u3.pkl
fi
$PY visualize.py --mode lowWe --lowwe-data data/lowWe_all --lowwe-ckpt "$LOWWE_CKPT"
