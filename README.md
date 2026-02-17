# QASTS
**Quantized Audio Spectrogram Transformer Soup: Robustness and Deployment Efficiency**

This repository provides the official implementation of **QASTS**, a quantized model soup approach
that improves **sampling-rate robustness** for Audio Spectrogram Transformer (AST) under QAT.

We gratefully acknowledge and build upon the codebases of **AST (Audio Spectrogram Transformer)** and  
**OFQ (Oscillation-free Quantization for Low-bit Vision Transformers)**.

---

## Supported Datasets
- **ESC-50**
- **FSD-Kaggle2018**

> Both datasets share the same script naming convention.

---

## Pipeline Overview (Stage 1–3)
All commands below are executed under `egs/`.

### 0) Data Preparation
```bash
cd <AST_ROOT>/egs/
bash run_prepare.sh
```

### 1) Stage 1 — Full-Precision Fine-tuning (FP)
```bash
bash run_fp.sh
```

### 2) Stage 2 — Quantization-Aware Training (QAT)
```bash
# Train baseline QAT
bash run_qat_baseline.sh

# Evaluate baseline QAT
bash run_qat_eval.sh
```

### 3) Stage 3 — Baselines and Ours
```bash
# Baseline: QAT+MR (single model, multi-rate training; 5 rates × 2 runs = 10 runs)
bash run_qat_mr.sh

# Baseline: Soup (no rate diversity; fixed 16 kHz)
bash run_soup_baseline.sh

# Ours: QASTS (rate diversity + frozen quantization scales)
# Rates: 12/16/20/24/28 kHz, 2 candidates per rate (total 10), frozen LSQ scale factors
bash run_qasts.sh
```

---

## Robustness Evaluation (Unseen Sampling Rates)
We evaluate robustness on unseen sampling rates: **8/14/22/30/44.1 kHz**.

```bash
# Example bit-widths (W/A)
bash run_robustness_eval.sh 4 4
bash run_robustness_eval.sh 4 8
bash run_robustness_eval.sh 8 8

# Or use qat_config.yml
bash run_robustness_eval.sh
```

---

## Notes
- Bit-widths are read from each experiment folder’s `qat_config.yml` (`wq_bitw`, `aq_bitw`).
- Robustness results are saved under each experiment folder: `robustness/`.
