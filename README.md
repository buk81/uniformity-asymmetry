# Uniformity Asymmetry: An Exploratory Metric for Detecting Representational Preferences in LLM Embeddings

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18110161.svg)](https://doi.org/10.5281/zenodo.18110161)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buk81/uniformity-asymmetry/blob/main/Uniformity_Asymmetry_Validation.ipynb)

**Author:** Davide D'Elia
**Date:** 2025-12-31
**Paper:** [arXiv link upon publication]

---

## Overview

This repository contains the code and dataset for the paper "Uniformity Asymmetry: An Exploratory Metric for Detecting Representational Preferences in LLM Embeddings".

**Uniformity Asymmetry** is an exploratory metric for detecting representational asymmetries in LLM embeddings—differences in how models cluster semantically equivalent statements with different framings.

> ⚠️ **Important Caveat:** Our dataset design introduces a structural confound: Side A statements are consistently more abstract/conceptual than Side B. This limits causal interpretation. Observed asymmetries may reflect representational compression rather than normative preferences. These findings are exploratory and require validation with balanced datasets.

## Key Features

- **Exploratory metric**: Detects representational asymmetries in embedding space
- **Cross-model validation**: Tested on 4 models (Gemma, Llama, Mistral, Apertus)
- **Publication-quality statistics**: 10,000 bootstrap resamples for confidence intervals
- **No fine-tuning required**: Works with any decoder model

## Files

```
uniformity-asymmetry/
├── README.md                              # This file
├── Uniformity_Asymmetry_Validation.ipynb  # Colab notebook (recommended)
├── uniformity_asymmetry_clean.py          # Standalone Python script
├── dataset.json                           # 230 statement pairs
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
├── .gitignore                             # Git ignore rules
├── uniformity_asymmetry_FINAL_DOI_*.tar.gz    # Archive (paper + code + data + DOI)
├── uniformity_asymmetry_FINAL_DOI_*.tar.gz.ots # Bitcoin timestamp proof
└── extended_results/                      # ⭐ NEW: Output correlation analysis
```

## Extended Results (v1.1)

**NEW (2026-01-01):** Post-paper validation experiments testing embedding→output correlation.

📁 **[extended_results/](extended_results/)** — Output Correlation Analysis

### Key Findings

| # | Discovery | Impact |
|---|-----------|--------|
| 1 | **Gemma r=0.95 is artifact** | Single category (GT-Numeric) drives entire correlation |
| 2 | **RLHF creates "Deceptive Alignment"** | Llama: neutral embeddings, biased outputs |
| 3 | **Multilingual inversion** | Apertus shows opposite pattern due to concept compression |
| 4 | **Lie vs Bullshit classification** | Categories cluster by confidence × asymmetry |

### Quick Links
- [FINDINGS_SUMMARY.md](extended_results/FINDINGS_SUMMARY.md) — Executive Summary
- [figures/](extended_results/figures/) — Visualizations
- [data/](extended_results/data/) — Raw analysis data

> These results **complement** the paper. The paper's cautious "exploratory" framing was correct.

## Quick Start (Google Colab)

1. Open `Uniformity_Asymmetry_Validation.ipynb` in Google Colab
2. Set your HuggingFace token in Colab Secrets:
   - Click the key icon (Secrets) in the left sidebar
   - Add `HF_TOKEN` with your token value
3. Change `MODEL_NAME` in Cell 2 to: `"gemma"`, `"llama"`, `"mistral"`, or `"apertus"`
4. Run all cells (Runtime > Run all)
5. Download the JSON results file

**Expected runtime:** ~25-35 minutes per model on A100

> **Note:** Free Tier Colab GPUs (T4) may be slower or run out of memory with 9B models. A100 or L4 recommended for reliable execution.

## Local Usage

```bash
# Install dependencies
pip install -r requirements.txt

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

## Results Summary (10,000 Bootstrap Resamples)

| Model | Lifestyle Δ | Scientific Δ | GT Numeric Δ | Neutral 95% CI | Status |
|-------|-------------|--------------|--------------|----------------|--------|
| Llama-3.1-8B | -0.008 | -0.013 | -0.142 | [-0.010, 0.015] | Neutral |
| Mistral-7B | -0.012 | +0.010 | -0.201 | [-0.015, 0.031] | Neutral |
| Gemma-2-9B | +0.021 | +0.021 | -0.251 | [0.013, 0.030] | Marginal |
| **Apertus-8B** | +0.036 | **+0.109** | **+0.097** | [0.030, 0.094] | **Asymmetry** |

### Key Observations

**Apertus-8B Shows Distinct Pattern:**
- Higher asymmetry in Scientific Facts (+0.109 vs ±0.02 in other models)
- Inverted structural effect on GT Numeric (+0.097 vs -0.14 to -0.25)
- Pattern consistent with preferring abstract/conceptual framings

**Interpretation (requires further validation):**
- These asymmetries may reflect representational compression (abstract concepts cluster more tightly) rather than normative preferences
- The structural confound (Side A = more abstract) limits causal claims
- Llama/Mistral show near-zero asymmetry on neutral categories → good calibration baseline

## Citation

```bibtex
@dataset{delia2025uniformity,
  author       = {D'Elia, Davide},
  title        = {{Uniformity Asymmetry: An Exploratory Metric for
                   Detecting Representational Preferences in LLM
                   Embeddings -- Code and Dataset}},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18110161},
  url          = {https://doi.org/10.5281/zenodo.18110161}
}
```

## License

MIT License

## Priority Proof (Bitcoin Timestamp)

This research is timestamped on the Bitcoin blockchain via [OpenTimestamps](https://opentimestamps.org/):

```
uniformity_asymmetry_FINAL_DOI_20251231_191728.tar.gz.ots
```

**Verify:**
```bash
pip install opentimestamps-client
ots verify uniformity_asymmetry_FINAL_DOI_20251231_191728.tar.gz.ots
```

The archive contains: paper source, code, dataset, and results—cryptographically proving existence as of 2025-12-31.

## Acknowledgments

- Compute resources: Google Colab (A100)
- Bitcoin blockchain timestamping: OpenTimestamps
- AI assistants used for Python coding routines and LaTeX formatting
