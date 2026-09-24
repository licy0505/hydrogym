#!/usr/bin/env bash
# End-to-end FNO / SDF / large-dataset experiment chain (CPU friendly, resumable).
#
#   bash run_experiments.sh            # everything
#   STEPS=1000 bash run_experiments.sh # quicker smoke run
#
# Each stage is skipped if its output already exists.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
STEPS=${STEPS:-3000}      # teacher-forced steps
USTEPS=${USTEPS:-400}     # unrolled fine-tune steps
mkdir -p logs ckpts results

# ---------------------------------------------------------------- data
for s in base large aug; do
  $PY generate_dataset.py --set $s --out data/$s --nsteps 2000 --ds 3 | tee -a logs/gen_$s.log
done

# ---------------------------------------------------------------- models
train() {  # name data extra-args...
  local name=$1 data=$2; shift 2
  [ -f ckpts/$name.pkl ] || $PY train_operator.py --data "$data" --steps "$STEPS" --out ckpts/$name.pkl "$@" 2>&1 | tee logs/train_$name.log
  [ -f ckpts/${name}_u3.pkl ] || $PY train_operator.py --data "$data" --resume ckpts/$name.pkl --unroll 3 --batch 8 \
      --lr 3e-4 --steps "$USTEPS" --out ckpts/${name}_u3.pkl 2>&1 | tee logs/train_${name}_u3.log
  for c in $name ${name}_u3; do
    [ -f results/$c.json ] || $PY evaluate_transfer.py --data data/base --ckpt ckpts/$c.pkl --json results/$c.json | tee logs/eval_$c.log
  done
}

[ -f results/persistence.json ] || $PY evaluate_transfer.py --data data/base --ckpt persistence --json results/persistence.json
train fno_sdf_large  data/base,data/large          --arch fno --geom sdf
train fno_chi_large  data/base,data/large          --arch fno --geom chi
train fno_sdf_aug    data/base,data/large,data/aug --arch fno --geom sdf

$PY compare_models.py
