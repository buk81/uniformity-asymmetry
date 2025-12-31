# Uniformity Asymmetry: Calibrated Detection of Normative Preferences in LLM Embeddings

**Author:** Davide D'Elia
**Date:** 2025-12-31
**Paper:** [arXiv link upon publication]

---

## Overview

This repository contains the code and dataset for the paper "Uniformity Asymmetry: Calibrated Detection of Normative Preferences in LLM Embeddings".

**Uniformity Asymmetry** is a calibrated metric for detecting hidden normative preferences in LLM embeddings - biases that remain invisible to generation-based safety filters.

## Key Features

- **Calibrated metric**: Converges to zero on genuinely neutral content (Lifestyle: Llama Δ = -0.008)
- **Cross-model validation**: Tested on 4 models (Gemma, Llama, Mistral, Apertus)
- **Publication-quality**: 10,000 bootstrap resamples for confidence intervals
- **No fine-tuning required**: Works with any decoder model

## Files

```
github_release/
├── README.md                              # This file
├── Uniformity_Asymmetry_Validation.ipynb  # Colab notebook (recommended)
├── uniformity_asymmetry_clean.py          # Standalone Python script
└── dataset.json                           # 230 statement pairs (optional)
```

## Quick Start (Google Colab)

1. Open `Uniformity_Asymmetry_Validation.ipynb` in Google Colab
2. Set your HuggingFace token in Colab Secrets:
   - Click the key icon (Secrets) in the left sidebar
   - Add `HF_TOKEN` with your token value
3. Change `MODEL_NAME` in Cell 2 to: `"gemma"`, `"llama"`, `"mistral"`, or `"apertus"`
4. Run all cells (Runtime > Run all)
5. Download the JSON results file

**Expected runtime:** ~25-35 minutes per model on A100

## Local Usage

```bash
# Set your HuggingFace token
export HF_TOKEN="your_token_here"

# Run for each model
python uniformity_asymmetry_clean.py --model gemma
python uniformity_asymmetry_clean.py --model llama
python uniformity_asymmetry_clean.py --model mistral
python uniformity_asymmetry_clean.py --model apertus
```

## Dataset

230 statement pairs across 6 categories:

| Category | Pairs | Purpose |
|----------|-------|---------|
| Ground Truth Numeric | 30 | Structural calibration |
| Ground Truth Non-Numeric | 20 | Factual equivalences |
| Tech Philosophy | 50 | Software "Holy Wars" |
| **Lifestyle** | **50** | **Calibration category** |
| Business | 50 | Organizational strategies |
| Scientific Facts | 30 | Scientific framings |

**Side A:** Centralized/Hierarchical/Conceptual framings
**Side B:** Decentralized/Autonomous/Numeric framings

## Method

1. Extract embeddings via mean pooling over last hidden layer (skip BOS)
2. Calculate uniformity score (average pairwise cosine similarity)
3. Compute asymmetry: Δ = U(Side A) - U(Side B)
4. Bootstrap 95% CI with 10,000 resamples
5. Cohen's d effect size

**Validation criteria:**
- 95% CI includes zero → PASS
- Cohen's d < 0.2 → Small effect

## Results Summary (Validated with 10,000 Bootstrap Resamples)

| Model | Lifestyle Δ | Scientific Δ | GT Numeric Δ | Neutral 95% CI | Status |
|-------|-------------|--------------|--------------|----------------|--------|
| Llama-3.1-8B | -0.008 | -0.013 | -0.142 | [-0.010, 0.015] | **PASS** |
| Mistral-7B | -0.012 | +0.010 | -0.201 | [-0.015, 0.031] | **PASS** |
| Gemma-2-9B | +0.021 | +0.021 | -0.251 | [0.013, 0.030] | REVIEW |
| **Apertus-8B** | +0.036 | **+0.109** | **+0.097** | [0.030, 0.094] | **DETECTED** |

### Key Findings

**Apertus-8B Anomaly:**
- Shows 5-10x higher asymmetry in Scientific Facts (+0.109 vs ±0.02)
- Inverted structural effect: prefers conceptual over numeric representations (+0.097 vs -0.14 to -0.25)
- Suggests epistemic preference for "concepts over data points"

**Calibration:**
- Llama and Mistral: CI includes zero, Cohen's d < 0.3 → neutral
- Gemma: marginal positive asymmetry (CI excludes zero but small effect)
- Apertus: clear systematic preference for Side A (hierarchical/conceptual)

## Citation

```bibtex
@article{delia2025uniformity,
  title={Uniformity Asymmetry: Calibrated Detection of Normative Preferences in LLM Embeddings},
  author={D'Elia, Davide},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

- Compute resources: Google Colab (A100)
- Bitcoin blockchain timestamping: OpenTimestamps
- AI assistants used for Python coding routines and LaTeX formatting
