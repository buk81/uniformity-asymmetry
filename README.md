# Uniformity Asymmetry & Phase-Structured Dynamics

[![DOI Paper 1](https://zenodo.org/badge/DOI/10.5281/zenodo.18110161.svg)](https://doi.org/10.5281/zenodo.18110161)
[![DOI Paper 2](https://zenodo.org/badge/DOI/10.5281/zenodo.18142454.svg)](https://doi.org/10.5281/zenodo.18142454)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buk81/uniformity-asymmetry/blob/main/notebooks/Uniformity_Asymmetry_Validation.ipynb)

**Author:** Davide D'Elia
**Release:** v2.0 (2026-01-04)

---

## Papers

| # | Title | DOI | Date |
|---|-------|-----|------|
| 1 | **Uniformity Asymmetry:** An Exploratory Metric for Detecting Representational Preferences in LLM Embeddings | [10.5281/zenodo.18110161](https://doi.org/10.5281/zenodo.18110161) | 2025-12-31 |
| 2 | **Layer-wise Embedding-Output Dynamics Across LLM Families:** Evidence for Phase-Structured Decision Commitment | [10.5281/zenodo.18142454](https://doi.org/10.5281/zenodo.18142454) | 2026-01-04 |

**Relation:** Paper #2 continues Paper #1, addressing two of three future research directions (output correlation, layer-wise analysis).

---

## Overview

This repository contains code, data, and reproducibility materials for two related papers on LLM embedding geometry.

**Paper #1 (Uniformity Asymmetry)** introduces an exploratory metric for detecting representational asymmetries in LLM embeddings—differences in how models cluster semantically equivalent statements with different framings.

**Paper #2 (Phase-Structured Dynamics)** extends this to layer-wise analysis across 4 model families (Pythia, Llama, Gemma, Apertus), demonstrating that embedding-output relationships exhibit **phase-structured dynamics**: early layers show positive correlation with output preference, while late layers show inversion.

> **Important Caveat:** Our dataset design introduces a structural confound: Side A statements are consistently more abstract/conceptual than Side B. This limits causal interpretation. These findings are exploratory.

---

## Key Findings (Paper #2)

| Model | Type | Early Layers | Late Layers | Phase-Structured? |
|-------|------|--------------|-------------|-------------------|
| Pythia-6.9B | Base | +0.44*** | **-0.17***| Yes |
| Llama-3.1-8B | Base | +0.05 | **-0.30*** | Yes |
| Apertus-8B | Multilingual | +0.39*** | **-0.25*** | Yes |
| Gemma-2B | SFT | +0.10 | -0.02 | No (boundary) |

**Central Finding:** Late-layer inversion is architecturally robust in larger base models (r = -0.17 to -0.41), but decision commitment depth is architecture-dependent.

---

## Repository Structure

```
uniformity-asymmetry/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── dataset.json                           # 230 statement pairs (shared)
│
├── paper2/                                # Paper #2: Phase-Structured Dynamics
│   ├── paper2_phase_structured_dynamics.tex
│   ├── references.bib
│   ├── Phase_Structured_Dynamics_DElia_2026.pdf
│   └── figures/
│       ├── fig1_main_results.png          # 4-panel cross-model figure
│       └── fig2_template_effect.png       # Template comparison
│
├── notebooks/                             # Colab-ready notebooks
│   ├── Uniformity_Asymmetry_Validation.ipynb   # Paper 1: Base metric
│   ├── Bootstrap_CI_Layer_Analysis.ipynb       # Paper 2: Pythia layer analysis
│   ├── Llama3_Cross_Model_Validation.ipynb     # Paper 2: Llama validation
│   ├── Llama3_Instruct_Template_Comparison.ipynb # Paper 2: Template effects
│   ├── Gemma_Cross_Model_Validation.ipynb      # Paper 2: Gemma (boundary)
│   ├── Apertus_Cross_Model_Validation.ipynb    # Paper 2: Apertus multilingual
│   └── Option_B_Pair_Level.ipynb               # Paper 2: Pair-level method
│
├── results/                               # JSON results + summaries
│   ├── pythia/                            # Pythia-6.9B, 12B results
│   ├── llama/                             # Llama-3.1-8B results
│   ├── gemma/                             # Gemma-2B results
│   ├── apertus/                           # Apertus-8B results
│   ├── CONSOLIDATED_RESULTS.csv           # All models summary
│   └── PHASE_SUMMARY.csv                  # Phase means per model
│
├── timestamps/                            # Bitcoin blockchain proofs
│   ├── paper1_20251231.tar.gz.ots
│   └── paper2_20260104.zip.ots
│
├── extended_results/                      # Post-paper validation (v1.1)
│   ├── FINDINGS_SUMMARY.md
│   ├── figures/
│   └── data/
│
└── uniformity_asymmetry_clean.py          # Standalone Python script
```

---

## Quick Start (Google Colab)

### Paper #1: Uniformity Asymmetry
1. Open `notebooks/Uniformity_Asymmetry_Validation.ipynb` in Colab
2. Add `HF_TOKEN` to Colab Secrets
3. Run all cells (~25-35 min on A100)

### Paper #2: Layer-wise Analysis
1. Open `notebooks/Bootstrap_CI_Layer_Analysis.ipynb` for Pythia
2. Or `notebooks/Llama3_Cross_Model_Validation.ipynb` for Llama
3. Results include layer-wise correlations with bootstrap CIs

---

## Dataset

230 statement pairs across 6 categories:

| Category | Pairs | Purpose |
|----------|-------|---------|
| Ground Truth Numeric | 30 | Structural calibration |
| Ground Truth Non-Numeric | 20 | Factual equivalences |
| Tech Philosophy | 50 | Software "Holy Wars" |
| Lifestyle | 50 | Calibration category |
| Business | 50 | Organizational strategies |
| Scientific Facts | 30 | Scientific framings |

**Side A:** Abstract/Conceptual framings
**Side B:** Specific/Numeric framings

---

## Citations

### Paper #1
```bibtex
@article{delia2025uniformity,
  author    = {D'Elia, Davide},
  title     = {Uniformity Asymmetry: An Exploratory Metric for Detecting
               Representational Preferences in {LLM} Embeddings},
  journal   = {Zenodo preprint},
  year      = {2025},
  doi       = {10.5281/zenodo.18110161}
}
```

### Paper #2
```bibtex
@article{delia2026phase,
  author    = {D'Elia, Davide},
  title     = {Layer-wise Embedding-Output Dynamics Across {LLM} Families:
               Evidence for Phase-Structured Decision Commitment},
  journal   = {Zenodo preprint},
  year      = {2026},
  doi       = {10.5281/zenodo.18142454}
}
```

---

## Priority Proof (Bitcoin Timestamps)

Both papers are timestamped on the Bitcoin blockchain via [OpenTimestamps](https://opentimestamps.org/):

| Paper | Timestamp File | Date |
|-------|----------------|------|
| #1 | `timestamps/paper1_20251231.tar.gz.ots` | 2025-12-31 |
| #2 | `timestamps/paper2_20260104.zip.ots` | 2026-01-04 |

**Verify:**
```bash
pip install opentimestamps-client
ots verify timestamps/paper1_20251231.tar.gz.ots
ots verify timestamps/paper2_20260104.zip.ots
```

---

## Post-Publication Updates

### Neutral Statement Control Test (2026-01-05)

Following community feedback (thanks Kevin!), we ran a control test comparing neutral vs political statement pairs:

| Category | Mean Asymmetry | Result |
|----------|----------------|--------|
| Neutral (n=12) | 0.045 | Higher! |
| Political (n=6) | 0.029 | Lower! |

**Key Finding:** The metric measures **embedding-space structural differences**, not "bias" per se. Political opposites are semantically CLOSE (same topic) → low asymmetry. Different factual statements can be semantically FAR → variable asymmetry.

**Full analysis:** `results/NEUTRAL_CONTROL_TEST_ANALYSIS.md`

---

## License

MIT License

## Acknowledgments

- Compute resources: Google Colab (A100)
- Bitcoin blockchain timestamping: OpenTimestamps
- AI assistants used for Python coding routines and LaTeX formatting

---

*Release v2.0 — 2026-01-04*
