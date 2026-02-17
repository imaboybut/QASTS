#!/bin/bash
set -e

source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

EPOCHS=${EPOCHS:-3}

for i in {0..9}; do
  python train_qasts.py --candidate_idx "$i" --num_epochs "${EPOCHS}" --rate_mode qasts
done
python train_qasts.py --mode soup --num_epochs "${EPOCHS}" --rate_mode qasts
