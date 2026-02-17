#!/bin/bash
set -e

source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

python evaluate_qat_baseline.py "$@"
