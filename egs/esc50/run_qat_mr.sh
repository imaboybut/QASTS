#!/bin/bash
set -e

source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

STAGES=${STAGES:-10}
EPOCHS=${EPOCHS:-8}
SEED=${SEED:-42}

if [ "$#" -ge 1 ]; then
  FOLDS=("$1")
else
  FOLDS=(1 2 3 4 5)
fi

for fold in "${FOLDS[@]}"; do
  python train_qat_mr.py --fold "${fold}" --stages "${STAGES}" --epochs "${EPOCHS}" --seed "${SEED}"
done
