#!/bin/bash
set -e

source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

STAGES=${STAGES:-10}
EPOCHS=${EPOCHS:-3}
SEED=${SEED:-42}

python train_qat_mr.py --stages "${STAGES}" --epochs "${EPOCHS}" --seed "${SEED}" "$@"
