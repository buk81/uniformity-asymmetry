# Transformer Dynamics Research Series

[![DOI Paper 1](https://zenodo.org/badge/DOI/10.5281/zenodo.18110161.svg)](https://doi.org/10.5281/zenodo.18110161)
[![DOI Paper 2](https://zenodo.org/badge/DOI/10.5281/zenodo.18142454.svg)](https://doi.org/10.5281/zenodo.18142454)
[![DOI Paper 3](https://zenodo.org/badge/DOI/10.5281/zenodo.18165365.svg)](https://doi.org/10.5281/zenodo.18165365)
[![DOI Paper 4](https://zenodo.org/badge/DOI/10.5281/zenodo.18316488.svg)](https://doi.org/10.5281/zenodo.18316488)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buk81/uniformity-asymmetry/blob/main/notebooks/Uniformity_Asymmetry_Validation.ipynb)

**Author:** Davide D'Elia
**Release:** v4.0 (2026-01-20)

---

## Papers

| # | Title | Status | Date |
|---|-------|--------|------|
| 1 | **Uniformity Asymmetry:** An Exploratory Metric for Detecting Representational Preferences in LLM Embeddings | [DOI: 10.5281/zenodo.18110161](https://doi.org/10.5281/zenodo.18110161) | 2025-12-31 |
| 2 | **Layer-wise Embedding-Output Dynamics Across LLM Families:** Evidence for Phase-Structured Decision Commitment | [DOI: 10.5281/zenodo.18142454](https://doi.org/10.5281/zenodo.18142454) | 2026-01-04 |
| 3 | **Thermodynamic Constraints in Transformer Architectures:** A Sheaf-Theoretic Perspective | [DOI: 10.5281/zenodo.18165365](https://doi.org/10.5281/zenodo.18165365) | 2026-01-06 |
| 4 | **Alignment Robustness Depends More on Training than Architecture:** A Cross-Vendor Analysis of Attention Specialization in Large Language Models | [DOI: 10.5281/zenodo.18316488](https://doi.org/10.5281/zenodo.18316488) | 2026-01-20 |

**Series Arc:**
- Paper 1 → Empirical observation (uniformity asymmetry)
- Paper 2 → Layer-wise dynamics (phase-structured commitment)
- Paper 3 → Theoretical consolidation (thermodynamic laws + sheaf theory)
- Paper 4 → **Cross-vendor validation** (alignment robustness hierarchy)

---

## Overview

This repository contains code, data, and reproducibility materials for four related papers on LLM embedding geometry and transformer dynamics.

**Paper 1 (Uniformity Asymmetry)** introduces an exploratory metric for detecting representational asymmetries in LLM embeddings—differences in how models cluster semantically equivalent statements with different framings.

**Paper 2 (Phase-Structured Dynamics)** extends this to layer-wise analysis across 4 model families (Pythia, Llama, Gemma, Apertus), demonstrating that embedding-output relationships exhibit **phase-structured dynamics**: early layers show positive correlation with output preference, while late layers show inversion.

**Paper 3 (Thermodynamic Constraints)** consolidates these observations into a theoretical framework using sheaf theory. It identifies three quantitative scaling laws governing transformer architectures, validated across 23+ models from 7 labs.

**Paper 4 (Alignment Robustness)** presents the first systematic cross-vendor study of how preference optimization (RLHF/DPO) affects attention head specialization. Testing 8 vendor families and 25+ model variants, it establishes a robustness hierarchy: **Training Methodology > SWA > Architecture > Scale**.

---

## Key Findings (Paper 4) — NEW

| Finding | Evidence |
|---------|----------|
| **SI Reduction Pattern** | RLHF/DPO reduces SI in unprotected architectures (LLaMA-3.1: −56.3%, LLaMA-2: −7.95%) |
| **SWA Protection** | Sliding Window Attention correlates with SI preservation (Mistral: +4.2%, Gemma-2: +1.9%) |
| **GQA Noise Sensitivity** | ~5,800× higher PPL-slope than MHA at matched scale (p < 0.01) |
| **Synthetic Immunity** | Phi family shows SI ≈ 0.33 invariant across 10.8× scale range |
| **Training Override** | Qwen2 (OMO training): No recursive degradation despite highest ρ_eff |

**Robustness Hierarchy:**
```
Training Methodology > Sliding Window Attention > Architecture (MHA/GQA) > Scale
```

**Perturbation Probe:** A diagnostic tool differentiating pathological (>20% SI response) from healthy low-SI states.

---

## Key Findings (Papers 1–3)

<details>
<summary>Paper 2: Phase-Structured Dynamics</summary>

| Model | Type | Early Layers | Late Layers | Phase-Structured? |
|-------|------|--------------|-------------|-------------------|
| Pythia-6.9B | Base | +0.44*** | **-0.17***| Yes |
| Llama-3.1-8B | Base | +0.05 | **-0.30*** | Yes |
| Apertus-8B | Multilingual | +0.39*** | **-0.25*** | Yes |
| Gemma-2B | SFT | +0.10 | -0.02 | No (boundary) |

**Central Finding:** Late-layer inversion is architecturally robust in larger base models.

</details>

<details>
<summary>Paper 3: Thermodynamic Constraints</summary>

| Law | Finding | Evidence |
|-----|---------|----------|
| **Kleiber's Law** | G_max = 10^(1/L) | r = -0.81, p = 0.014 (Pythia family) |
| **Training Heritage** | Lab determines thermodynamic sign | EleutherAI: 80% dampening vs. Meta/OpenAI: 100% expansion |
| **Spectral Signature** | ‖W_V‖/‖W_O‖ ratio predicts behavior | 10× magnitude differences between labs |

**Core Finding:** The hierarchy is **Heritage > Geometry > Scale**.

</details>

---

## Repository Structure

```
uniformity-asymmetry/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── dataset.json                           # 230 statement pairs (Papers 1-2)
│
├── paper2/                                # Paper 2: Phase-Structured Dynamics
│   ├── Phase_Structured_Dynamics_DElia_2026.pdf
│   └── ...
│
├── paper3/                                # Paper 3: Thermodynamic Constraints
│   ├── Thermodynamic_Constraints_DElia_2026.pdf
│   ├── README.md
│   ├── notebooks/                         # 18 core experiment notebooks
│   └── Results/
│
├── paper4/                                # Paper 4: Alignment Robustness [NEW]
│   ├── Alignment_Robustness_...pdf        # Final paper
│   ├── README.md                          # Reproducibility guide
│   ├── claims.yaml                        # Machine-readable claim mapping
│   ├── verify_release.py                  # Package integrity checker
│   ├── reproduce.py                       # CLI reproduction tool
│   ├── prompts/
│   │   └── standard10_v3.txt              # Standard-10 prompts (SHA256 verified)
│   ├── notebooks/                         # 19 A-Tier notebooks
│   │   ├── A1_territorial/  (6)           # Territorial Collapse
│   │   ├── A2_indra/        (6)           # Indra State-Dependency
│   │   ├── A3_heritage/     (3)           # Heritage > Scale
│   │   ├── A4_synthetic/    (3)           # Synthetic Immunity
│   │   └── A5_mqa/          (1)           # MQA Pre-Collapsed
│   ├── results/                           # 27 JSON result files
│   │   ├── schema_v1.json
│   │   ├── E11/, E04/, E06/, E08/, E21_E22/, E-ISO/
│   └── src/
│       └── metrics.py                     # SI/PPL computation
│
├── notebooks/                             # Papers 1-2 notebooks
├── Results/                               # Papers 1-2 results
├── timestamps/                            # Bitcoin blockchain proofs
└── extended_results/                      # Post-paper validation
```

---

## Quick Start

### Paper 4: Alignment Robustness (NEW)

```bash
cd paper4

# 1. Verify package integrity
python verify_release.py

# 2. Reproduce E11 experiment (requires A100 GPU)
python reproduce.py --experiment E11 --model mistral --seed 42

# 3. Or use notebooks in Google Colab
# Upload any notebook from paper4/notebooks/ to Colab (A100 recommended)
```

**A-Tier Claims:**

| Claim | Notebooks | Key Metric |
|-------|-----------|------------|
| A1: Territorial Collapse | `A1_territorial/` | ΔSI varies by architecture |
| A2: Indra State-Dependency | `A2_indra/` | +114% heal / -7.8% damage |
| A3: Heritage > Scale | `A3_heritage/` | RLHF fragility pattern |
| A4: Synthetic Immunity | `A4_synthetic/` | Phi SI ≈ 0.33 |
| A5: MQA Pre-Collapsed | `A5_mqa/` | Falcon SI ≈ 0.14 |

### Papers 1-3

<details>
<summary>Paper 1: Uniformity Asymmetry</summary>

1. Open `notebooks/Uniformity_Asymmetry_Validation.ipynb` in Colab
2. Add `HF_TOKEN` to Colab Secrets
3. Run all cells (~25-35 min on A100)

</details>

<details>
<summary>Paper 2: Layer-wise Analysis</summary>

1. Open `notebooks/Bootstrap_CI_Layer_Analysis.ipynb` for Pythia
2. Or `notebooks/Llama3_Cross_Model_Validation.ipynb` for Llama
3. Results include layer-wise correlations with bootstrap CIs

</details>

<details>
<summary>Paper 3: Thermodynamic Constraints</summary>

1. Start with [`paper3/ELI5.md`](paper3/ELI5.md) for an intuitive explanation
2. See `paper3/README.md` for detailed notebook → claim mapping
3. All experiments use `PYTHONHASHSEED=42` for reproducibility

</details>

---

## Citations

### Paper 4 (NEW)
```bibtex
@misc{delia2026alignment,
  author    = {D'Elia, Davide},
  title     = {Alignment Robustness Depends More on Training than Architecture:
               A Cross-Vendor Analysis of Attention Specialization in Large Language Models},
  journal   = {Zenodo preprint},
  year      = {2026},
  doi       = {10.5281/zenodo.18316488}
}
```

### Paper 3
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

### Paper 2
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

### Paper 1
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

---

## Priority Proof (Bitcoin Timestamps)

All papers are timestamped on the Bitcoin blockchain via [OpenTimestamps](https://opentimestamps.org/):

| Paper | Timestamp File | Date |
|-------|----------------|------|
| 1 | `timestamps/paper1_20251231.tar.gz.ots` | 2025-12-31 |
| 2 | `timestamps/paper2_20260104.zip.ots` | 2026-01-04 |
| 3 | `timestamps/paper3_sheaf_theory_20260104.tar.gz.ots` | 2026-01-04 |
| 4 | *TBD after acceptance* | 2026-01-20 |

**Verify:**
```bash
pip install opentimestamps-client
ots verify timestamps/paper1_20251231.tar.gz.ots
```

---

## Dataset (Papers 1-2)

230 statement pairs across 6 categories:

| Category | Pairs | Purpose |
|----------|-------|---------|
| Ground Truth Numeric | 30 | Structural calibration |
| Ground Truth Non-Numeric | 20 | Factual equivalences |
| Tech Philosophy | 50 | Software "Holy Wars" |
| Lifestyle | 50 | Calibration category |
| Business | 50 | Organizational strategies |
| Scientific Facts | 30 | Scientific framings |

---

## License

MIT License

## Acknowledgments

- Compute resources: Google Colab (A100)
- Bitcoin blockchain timestamping: OpenTimestamps
- AI assistants used for Python coding routines and LaTeX formatting

---

*Release v4.0 — 2026-01-20*
