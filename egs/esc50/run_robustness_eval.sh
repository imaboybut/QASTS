#!/bin/bash
set -e

source ../../venvast/bin/activate
export TORCH_HOME=../../pretrained_models

if [ "$#" -eq 3 ]; then
    BIT_W=$1
    BIT_A=$2
    FOLD=$3
    python evaluate_robustness.py --bit_w "$BIT_W" --bit_a "$BIT_A" --fold "$FOLD"
elif [ "$#" -eq 2 ]; then
    BIT_W=$1
    BIT_A=$2
    python evaluate_robustness.py --bit_w "$BIT_W" --bit_a "$BIT_A"
elif [ "$#" -eq 0 ]; then
    python evaluate_robustness.py
else
    echo "Usage: $0 [bit_w bit_a [fold]]"
    exit 1
fi
