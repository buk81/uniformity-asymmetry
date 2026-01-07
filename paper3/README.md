# Paper 3: Thermodynamic Constraints in Transformer Architectures

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18165365.svg)](https://doi.org/10.5281/zenodo.18165365)

**A Sheaf-Theoretic Perspective**

This folder contains all materials for Paper #3 in the Transformer Dynamics Research Series.

---

## Quick Links

| Resource | Location |
|----------|----------|
| **PDF** | `Thermodynamic_Constraints_DElia_2026.pdf` |
| **Figures** | `Figures/` (9 figures: Fig 1-3 + A1-A5 + combined) |
| **Notebooks** | `notebooks/` (18 core experiments) |
| **Results** | `Results/` (JSONs, PNGs) |
| **Artifact Map** | `ARTIFACT_MAP.md` |

---

## Key Findings

| Law | Finding | Evidence |
|-----|---------|----------|
| **Kleiber's Law** | G_max = 10^(1/L) | r = -0.81, p = 0.014 |
| **Training Heritage** | Lab determines thermodynamic sign | EleutherAI: dampening, Meta/OpenAI: expansion |
| **Spectral Signature** | ‖W_V‖/‖W_O‖ ratio predicts behavior | 10× magnitude differences |

**Core Finding:** Heritage > Geometry > Scale

---

## Paper Claims → Notebooks

This table maps each major claim in the paper to the notebook(s) that validate it.

| Paper Section | Claim | Notebook(s) |
|---------------|-------|-------------|
| **§4 (Fig. 1)** | Kleiber's Law: G_max = 10^(1/L) | `Scaling_Law_Multi_Pythia.ipynb` |
| **§5.1 (Fig. 2)** | Cross-architecture benchmark (23+ models) | `Grand_Unified_Thermodynamic_Benchmark.ipynb`, `4Model_Cross_Architecture_Validation.ipynb` |
| **§5.2** | Training Heritage dominates | `Grand_Unified_Thermodynamic_Benchmark.ipynb` |
| **§5.3 (Fig. 3)** | Spectral Signature: ‖W_V‖/‖W_O‖ | `Restriction_Maps_Extraction.ipynb` |
| **§5.4** | Dimensional Crowding (ρ = H/d_head) | `High_Rho_Model_Hunt_NO_FINAL_LN.ipynb` |
| **§5.5** | RLHF Invariance (Twin Test) | `Twin_Test_Base_vs_Instruct.ipynb` |
| **§5.6** | Mistral Paradox | `Mistral_Paradox_Investigation.ipynb` |
| **Appendix A** | OPT Anomaly Investigation | `OPT_Anomaly_Investigation.ipynb` |
| **Appendix B (Fig. A5)** | Sheaf Laplacian Trace Proxy | `Restriction_Maps_Extraction.ipynb`, `H4_v2_Extended_Models.ipynb` |
| **Fig. A1** | Layer Dynamics | `Clean_Residual_Gain_NO_FINAL_LN.ipynb` |
| **Fig. A2** | L* Validation | `L_Star_Cross_Heritage_SignChange.ipynb` |
| **Fig. A3** | Pythia Scaling | `Pythia_Family_Residual_Gain_Sweep.ipynb` |
| **Fig. A4** | Input Robustness | `Input_Dependency_Thermodynamics.ipynb` |

---

## All 18 Core Notebooks

| Category | Notebooks |
|----------|-----------|
| **Core Validation** | `4Model_Cross_Architecture_Validation`, `Grand_Unified_Thermodynamic_Benchmark` |
| **Scaling Laws** | `Scaling_Law_Multi_Pythia`, `Pythia_Family_Residual_Gain_Sweep`, `High_Rho_Model_Hunt_NO_FINAL_LN` |
| **Sheaf Theory** | `Restriction_Maps_Extraction`, `H4_v2_Extended_Models` |
| **Architecture Tests** | `GPT2_LayerNorm_Validation`, `GPTJ_Parallel_Architecture_Test`, `FFN_Expansion_Analysis` |
| **Anomaly Investigations** | `OPT_Anomaly_Investigation`, `Mistral_Paradox_Investigation`, `Hypothesis_Tests_LLaMA_Anomaly` |
| **Twin Tests** | `Twin_Test_Base_vs_Instruct` |
| **L* Formula** | `L_Star_Cross_Heritage_SignChange` |
| **Robustness** | `Input_Dependency_Thermodynamics`, `Clean_Residual_Gain_NO_FINAL_LN` |
| **Special** | `Bentov_Point_Characterization` |

---

## Reproduction

### Prerequisites
```bash
pip install torch transformers numpy scipy matplotlib seaborn
```

### Quick Start (5 Core Notebooks)

1. **`Scaling_Law_Multi_Pythia.ipynb`** → Kleiber's Law (Fig. 1)
2. **`Grand_Unified_Thermodynamic_Benchmark.ipynb`** → Training Heritage (Fig. 2)
3. **`Restriction_Maps_Extraction.ipynb`** → Spectral Signature (Fig. 3)
4. **`Twin_Test_Base_vs_Instruct.ipynb`** → RLHF Invariance (§5.5)
5. **`High_Rho_Model_Hunt_NO_FINAL_LN.ipynb`** → Dimensional Crowding (§5.4)

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
| `high_rho_hunt_NO_FINAL_LN_*.json` | High_Rho_Model_Hunt_NO_FINAL_LN | §5.4 |

---

## Structure

```
paper3/
├── Thermodynamic_Constraints_DElia_2026.pdf  # Final paper
├── README.md                                  # This file
├── ARTIFACT_MAP.md                            # Artifact ↔ Notebook mapping
│
├── Figures/                                   # Paper figures (9 total)
│   ├── fig1_kleiber_law.png
│   ├── fig2_training_heritage.png
│   ├── fig3_spectral_signature.png
│   ├── fig_a1-a5_*.png                        # Appendix figures
│   └── fig_combined.png                       # All figures combined
│
├── notebooks/                                 # Experiment notebooks
│   ├── 4Model_Cross_Architecture_Validation.ipynb
│   ├── Scaling_Law_Multi_Pythia.ipynb
│   ├── Twin_Test_Base_vs_Instruct.ipynb
│   └── ... (18 core notebooks)
│
└── Results/                                   # JSON results + plots
    ├── *.json                                 # Raw measurements
    └── *.png                                  # Generated figures
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
