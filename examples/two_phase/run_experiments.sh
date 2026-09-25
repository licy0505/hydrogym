#!/usr/bin/env bash
# End-to-end FNO / SDF / validated-data experiment chain.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
STEPS=${STEPS:-3000}
USTEPS=${USTEPS:-400}
mkdir -p logs ckpts results

# Generator auto-regenerates stale schema files.
for s in base large aug lowWe_all; do
  $PY generate_dataset.py --set "$s" --out "data/$s" --nsteps 2000 --ds 3 | tee -a "logs/gen_$s.log"
done

ckpt_current() {
  local f=$1
  [ -f "$f" ] || return 1
  $PY - "$f" <<'PY'
import pickle, sys
try:
    with open(sys.argv[1], "rb") as fh:
        ck = pickle.load(fh)
    cfg = ck.get("cfg") or {}
    ok = int(cfg.get("dataset_schema_version", 0)) == 2
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

train() {
  local name=$1 data=$2 eval_data=$3; shift 3
  if ! ckpt_current "ckpts/$name.pkl"; then
    rm -f "ckpts/$name.pkl" "ckpts/${name}_u3.pkl"
    $PY train_operator.py --data "$data" --steps "$STEPS" --out "ckpts/$name.pkl" "$@" \
      2>&1 | tee "logs/train_$name.log"
  fi
  if ! ckpt_current "ckpts/${name}_u3.pkl"; then
    $PY train_operator.py --data "$data" --resume "ckpts/$name.pkl" --unroll 3 --batch 8 \
      --lr 3e-4 --steps "$USTEPS" --out "ckpts/${name}_u3.pkl" \
      2>&1 | tee "logs/train_${name}_u3.log"
  fi
  for c in "$name" "${name}_u3"; do
    $PY evaluate_transfer.py --data "$eval_data" --ckpt "ckpts/$c.pkl" \
      --json "results/$c.json" | tee "logs/eval_$c.log"
  done
}

$PY evaluate_transfer.py --data data/base --ckpt persistence --json results/persistence.json

train fno_sdf_large  data/base,data/large          data/base --arch fno --geom sdf
train fno_chi_large  data/base,data/large          data/base --arch fno --geom chi
train fno_sdf_aug    data/base,data/large,data/aug data/base --arch fno --geom sdf

# Dedicated low-We model: train only on the simple low-We train split, evaluate
# strictly on the complex low-We test split in the same dataset directory.
train lowWe_fno_sdf data/lowWe_all data/lowWe_all --families simple --arch fno --geom sdf

$PY compare_models.py
$PY visualize.py --mode lowWe \
  --lowwe-data data/lowWe_all \
  --lowwe-ckpt ckpts/lowWe_fno_sdf_u3.pkl
