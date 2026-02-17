# Quantized Audio Spectrogram Transformer Soup: Robustness and Deployment Efficiency
# QASTS

We gratefully acknowledge and reference the codebases of AST: Audio Spectrogram Transformer and OFQ: Oscillation-free Quantization for Low-bit Vision Transformers.



ESC-50, FSD-Kaggle2018
==============================

1) Data preparation
cd <AST_ROOT>/egs/~
bash run_prepare.sh

2) Stage 1: FP training (5-fold)
bash run_fp.sh

3) Stage 2: QAT training (5-fold)
bash run_qat_baseline.sh

4) Stage 2: QAT evaluation
bash run_qat_eval.sh

5) Baseline - QAT+MR (single model, 10 stages = 5 rates x 2)
bash run_qat_mr.sh

6) Baseline - Soup (no rate diversity, fixed 16k)
bash run_soup_baseline.sh

7) Ours - QASTS (rate diversity + frozen scales)
# 12/16/20/24/28kHz, 2 candidates per rate (total 10), fixed scale factors
bash run_qasts.sh

8) Robustness comparison (unseen rates: 8/14/22/30/44.1kHz)
# Example bit configs
bash run_robustness_eval.sh 4 4
bash run_robustness_eval.sh 4 8
bash run_robustness_eval.sh 8 8
# Or use qat_config.yml
bash run_robustness_eval.sh


[Notes]
- ESC-50 and FSD-Kaggle2018 use the same script naming convention.
- Bit settings are read from each folder's qat_config.yml (wq_bitw, aq_bitw).
- Robustness results are saved under each folder's robustness/ directory.
