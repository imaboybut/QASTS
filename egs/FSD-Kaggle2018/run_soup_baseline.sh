#!/bin/bash
set -e

source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

EPOCHS=${EPOCHS:-3}

for i in {0..9}; do
  python train_soup_baseline.py --candidate_idx "$i" --num_epochs "${EPOCHS}"
done
python train_soup_baseline.py --mode soup --num_epochs "${EPOCHS}"
