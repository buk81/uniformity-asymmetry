# Paper 3: Thermodynamic Constraints in Transformer Architectures

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18165365.svg)](https://doi.org/10.5281/zenodo.18165365)

**A Sheaf-Theoretic Perspective**

This folder contains all materials for Paper #3 in the Transformer Dynamics Research Series.

---

## Quick Links

| Resource | Location |
|----------|----------|
| **PDF** | `Thermodynamic_Constraints_DElia_2026.pdf` |
| **LaTeX** | `latex/thermodynamic_constraints.tex` |
| **Figures** | `Figures/` (9 PNGs) |
| **Notebooks** | `notebooks/` (40+ experiments) |
| **Results** | `Results/` (JSONs, PNGs) |
| **Artifact Map** | `ARTIFACT_MAP.md` |

---

## Key Findings

| Law | Finding | Evidence |
|-----|---------|----------|
| **Kleiber's Law** | G_max = 10^(1/L) | r = -0.81, p = 0.014 |
| **Training Heritage** | Lab determines thermodynamic sign | EleutherAI: dampening, Meta/OpenAI: expansion |
| **Spectral Signature** | ||W_V||/||W_O|| predicts behavior | 10× magnitude differences |

**Core Finding:** Heritage > Geometry > Scale

---

## Reproduction

### Prerequisites
```bash
pip install torch transformers numpy scipy matplotlib seaborn
```

### Core Notebooks (in recommended order)

1. **Cross-Architecture Validation**
   - `notebooks/4Model_Cross_Architecture_Validation.ipynb`
   - Validates thermodynamic signatures across model families

2. **Pythia Scaling Law**
   - `notebooks/Scaling_Law_Multi_Pythia.ipynb`
   - Kleiber's Law derivation (Fig. 1)

3. **Twin Test (Base vs Instruct)**
   - `notebooks/Twin_Test_Base_vs_Instruct.ipynb`
   - Thermodynamic invariance under RLHF

4. **Sheaf Laplacian**
   - `notebooks/Restriction_Maps_Extraction.ipynb`
   - Trace proxy computation (Fig. A5)

### Reproducibility
All experiments use `PYTHONHASHSEED=42` for deterministic results.

---

## Artifact Map

See `ARTIFACT_MAP.md` for complete mapping of:
- Which notebook produces which JSON/PNG
- Status: **core** (in paper), **obsolet**, **rejected**

### Core Artifacts (subset)

| Artifact | Notebook | Figure |
|----------|----------|--------|
| `thermodynamic_benchmark_*.json` | Grand_Unified_Thermodynamic_Benchmark | Fig. 2 |
| `scaling_law_multi_pythia_*.json` | Scaling_Law_Multi_Pythia | Fig. 1 |
| `twin_test_results_*.json` | Twin_Test_Base_vs_Instruct | §5.5 |
| `restriction_maps_results.json` | Restriction_Maps_Extraction | Fig. A5 |

---

## Structure

```
paper3/
├── Thermodynamic_Constraints_DElia_2026.pdf  # Final paper
├── README.md                                  # This file
├── ARTIFACT_MAP.md                            # Artifact ↔ Notebook mapping
│
├── latex/                                     # LaTeX sources
│   ├── thermodynamic_constraints.tex
│   ├── references.bib
│   └── neurips_2024.sty
│
├── Figures/                                   # Paper figures
│   ├── fig1_kleiber_law.png
│   ├── fig2_training_heritage.png
│   ├── fig3_spectral_signature.png
│   └── fig_a1-a5_*.png
│
├── notebooks/                                 # Experiment notebooks
│   ├── 4Model_Cross_Architecture_Validation.ipynb
│   ├── Scaling_Law_Multi_Pythia.ipynb
│   ├── Twin_Test_Base_vs_Instruct.ipynb
│   └── ... (40+ notebooks)
│
├── Results/                                   # JSON results + plots
│   ├── *.json                                 # Raw measurements
│   └── *.png                                  # Generated figures
│
└── timestamps/                                # OpenTimestamps proofs
```

---

## Citation

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

## Related Papers

| # | Title | DOI |
|---|-------|-----|
| 1 | Uniformity Asymmetry | [10.5281/zenodo.18110161](https://doi.org/10.5281/zenodo.18110161) |
| 2 | Phase-Structured Dynamics | [10.5281/zenodo.18142454](https://doi.org/10.5281/zenodo.18142454) |

---

*Part of the [Transformer Dynamics Research Series](../README.md)*
