# Transformer Dynamics Research Series

[![DOI Paper 1](https://zenodo.org/badge/DOI/10.5281/zenodo.18110161.svg)](https://doi.org/10.5281/zenodo.18110161)
[![DOI Paper 2](https://zenodo.org/badge/DOI/10.5281/zenodo.18142454.svg)](https://doi.org/10.5281/zenodo.18142454)
[![DOI Paper 3](https://zenodo.org/badge/DOI/10.5281/zenodo.18165365.svg)](https://doi.org/10.5281/zenodo.18165365)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buk81/uniformity-asymmetry/blob/main/notebooks/Uniformity_Asymmetry_Validation.ipynb)

**Author:** Davide D'Elia
**Release:** v3.0 (2026-01-06)

---

## Papers

| # | Title | DOI | Date |
|---|-------|-----|------|
| 1 | **Uniformity Asymmetry:** An Exploratory Metric for Detecting Representational Preferences in LLM Embeddings | [10.5281/zenodo.18110161](https://doi.org/10.5281/zenodo.18110161) | 2025-12-31 |
| 2 | **Layer-wise Embedding-Output Dynamics Across LLM Families:** Evidence for Phase-Structured Decision Commitment | [10.5281/zenodo.18142454](https://doi.org/10.5281/zenodo.18142454) | 2026-01-04 |
| 3 | **Thermodynamic Constraints in Transformer Architectures:** A Sheaf-Theoretic Perspective | [10.5281/zenodo.18165365](https://doi.org/10.5281/zenodo.18165365) | 2026-01-06 |

**Series Arc:**
- Paper #1 → Empirical observation (uniformity asymmetry)
- Paper #2 → Layer-wise dynamics (phase-structured commitment)
- Paper #3 → **Theoretical consolidation** (thermodynamic laws + sheaf theory)

---

## Overview

This repository contains code, data, and reproducibility materials for three related papers on LLM embedding geometry and transformer dynamics.

**Paper #1 (Uniformity Asymmetry)** introduces an exploratory metric for detecting representational asymmetries in LLM embeddings—differences in how models cluster semantically equivalent statements with different framings.

**Paper #2 (Phase-Structured Dynamics)** extends this to layer-wise analysis across 4 model families (Pythia, Llama, Gemma, Apertus), demonstrating that embedding-output relationships exhibit **phase-structured dynamics**: early layers show positive correlation with output preference, while late layers show inversion.

**Paper #3 (Thermodynamic Constraints)** consolidates these observations into a theoretical framework using sheaf theory. It identifies three quantitative scaling laws governing transformer architectures, validated across 23+ models from 7 labs.

---

## Key Findings (Papers #1–2)

> **Paper #1 Caveat:** Our dataset design introduces a structural confound: Side A statements are consistently more abstract/conceptual than Side B. This limits causal interpretation of the uniformity asymmetry metric. Paper #2's phase-structured findings are validated across independent model families.

### Paper #2 Results

| Model | Type | Early Layers | Late Layers | Phase-Structured? |
|-------|------|--------------|-------------|-------------------|
| Pythia-6.9B | Base | +0.44*** | **-0.17***| Yes |
| Llama-3.1-8B | Base | +0.05 | **-0.30*** | Yes |
| Apertus-8B | Multilingual | +0.39*** | **-0.25*** | Yes |
| Gemma-2B | SFT | +0.10 | -0.02 | No (boundary) |

**Central Finding:** Late-layer inversion is architecturally robust in larger base models (r = -0.17 to -0.41), but decision commitment depth is architecture-dependent.

---

## Key Findings (Paper #3)

| Law | Finding | Evidence |
|-----|---------|----------|
| **Kleiber's Law** | G_max = 10^(1/L) | r = -0.81, p = 0.014 (Pythia family) |
| **Training Heritage** | Lab determines thermodynamic sign | EleutherAI: 80% dampening vs. Meta/OpenAI: 100% expansion |
| **Spectral Signature** | ||W_V||/||W_O|| predicts behavior | 10× magnitude differences between labs |

**Additional Contributions:**
- **Sheaf Laplacian Validation:** GPT-2 exhibits 26× higher trace proxy than OPT-125m
- **Dimensional Crowding:** Head density ρ = H/d_head explains the Pythia anomaly
- **Thermodynamic Invariance:** RLHF cannot invert sign, only modulate magnitude

**Core Finding:** The hierarchy is **Heritage > Geometry > Scale**.

---

## Repository Structure

```
uniformity-asymmetry/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── dataset.json                           # 230 statement pairs (Papers 1-2)
│
├── paper2/                                # Paper #2: Phase-Structured Dynamics
│   ├── Phase_Structured_Dynamics_DElia_2026.pdf
│   ├── paper2_phase_structured_dynamics.tex
│   └── references.bib
│
├── paper3/                                # Paper #3: Thermodynamic Constraints
│   ├── Thermodynamic_Constraints_DElia_2026.pdf
│   ├── README.md                          # Paper 3 guide
│   ├── ARTIFACT_MAP.md                    # Artifact ↔ Notebook mapping
│   ├── latex/                             # LaTeX sources
│   ├── Figures/                           # Main paper figures
│   ├── notebooks/                         # 33 experiment notebooks
│   └── Results/                           # JSONs, PNGs (see ARTIFACT_MAP.md)
│
├── notebooks/                             # Papers 1-2 notebooks
│   ├── Uniformity_Asymmetry_Validation.ipynb
│   ├── Bootstrap_CI_Layer_Analysis.ipynb
│   └── ...
│
├── results/                               # Papers 1-2 results
│   ├── pythia/, llama/, gemma/, apertus/
│   └── CONSOLIDATED_RESULTS.csv
│
├── timestamps/                            # Bitcoin blockchain proofs
│   ├── paper1_20251231.tar.gz.ots
│   ├── paper2_20260104.zip.ots
│   └── paper3_20260106.tar.gz.ots
│
└── extended_results/                      # Post-paper validation
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

### Paper #3
```bibtex
@article{delia2026thermodynamic,
  author    = {D'Elia, Davide},
  title     = {Thermodynamic Constraints in Transformer Architectures:
               A Sheaf-Theoretic Perspective},
  journal   = {Zenodo preprint},
  year      = {2026},
  doi       = {10.5281/zenodo.18165365}
}
```

---

## Priority Proof (Bitcoin Timestamps)

All papers are timestamped on the Bitcoin blockchain via [OpenTimestamps](https://opentimestamps.org/):

| Paper | Timestamp File | Date |
|-------|----------------|------|
| #1 | `timestamps/paper1_20251231.tar.gz.ots` | 2025-12-31 |
| #2 | `timestamps/paper2_20260104.zip.ots` | 2026-01-04 |
| #3 | `paper3/timestamps/*.ots` | 2026-01-06 |

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

*Release v3.0 — 2026-01-06*
