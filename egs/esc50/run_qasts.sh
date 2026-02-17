#!/bin/bash
set -e

source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

if [ "$#" -eq 0 ]; then
  python train_qasts.py --fold 0 --num_candidates 10
else
  python train_qasts.py "$@"
fi
