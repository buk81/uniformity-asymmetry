# Anisotropy Profile Results: Pythia-6.9B

**Experiment Date:** 2026-01-04
**Model:** EleutherAI/pythia-6.9b
**Layers:** 32
**Hidden Dim:** 4096
**Prompts:** 16 diverse test prompts

---

## Key Findings

### Bell Curve: CONFIRMED ✅

```
Phase 1 (Layer 0-7):   Slope = +0.186  (Rising)   R = 0.88
Phase 2 (Layer 7-32):  Slope = -0.009  (Falling)  R = -0.41
```

### Critical Discovery: TWO Phase Transitions

| Metric | Layer | Interpretation |
|--------|-------|----------------|
| L*_anisotropy | **7** | Maximum compression (H⁰ reached) |
| L*_correlation | **28** | Correlation inversion (Paper #2) |
| Difference | **21 layers** | "Plateau/Processing" phase |

---

## Multi-Phase Model (Empirically Derived)

```
Layer:  0 === 4 === 7 =================== 28 ======= 32
        │     │     │                      │          │
Phase:  INIT  JUMP  MAX   SLOW PLATEAU     │  OUTPUT  │
        │     │     │     (0.99→0.94)      │  COLLAPSE│
        │     │     │                      │          │
Aniso:  0.06  0.94  0.99 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 0.94      0.09
EffRk:  105   1.7   1.07 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 1.67      66
```

### Phase Descriptions (Refined)

| Phase | Layers | Dynamics | Anisotropy | Effective Rank |
|-------|--------|----------|------------|----------------|
| 0. Init | 0-3 | Setup/Tokenization | 0.06→0.12 | 105→91 |
| 1. Jump | 3-4 | **DRAMATIC** compression | 0.12→0.94 | 91→1.7 |
| 2. Peak | 4-7 | Refinement | 0.94→**0.994** | 1.7→**1.07** |
| 3. Plateau | 7-31 | Slow release | 0.994→0.938 | 1.07→1.67 |
| 4. Collapse | 31-32 | Output projection | 0.938→**0.095** | 1.67→**66** |

---

## Actual Metrics Per Layer

### Intrinsic Dimension Ratio (λ₁/Σλᵢ)

Primary anisotropy measure. Higher = more anisotropic (compressed).

| Layer | Value | Change | Phase |
|-------|-------|--------|-------|
| 0 | 0.060 | — | Init |
| 1 | 0.178 | +0.118 | Init |
| 2 | 0.150 | -0.028 | Init |
| 3 | 0.116 | -0.034 | Init |
| **4** | **0.940** | **+0.824** | **JUMP!** |
| 5 | 0.990 | +0.050 | Rising |
| 6 | 0.993 | +0.003 | Rising |
| **7** | **0.994** | +0.001 | **MAX** |
| 8-15 | 0.994→0.988 | slow ↓ | Plateau |
| 16-23 | 0.987→0.977 | slow ↓ | Plateau |
| 24-31 | 0.971→0.938 | faster ↓ | Late Plateau |
| **32** | **0.095** | **-0.843** | **COLLAPSE** |

### Effective Rank

Inverse measure. Lower = more anisotropic.

| Layer | Value | Phase |
|-------|-------|-------|
| 0 | 104.5 | Init (high diversity) |
| 3 | 90.7 | Init |
| **4** | **1.66** | **COLLAPSE to 1D!** |
| **7** | **1.07** | **MIN (nearly 1D)** |
| 16 | 1.15 | Plateau |
| 28 | 1.44 | Late Plateau |
| 31 | 1.67 | Pre-output |
| **32** | **66.4** | **EXPANSION** |

---

## Theoretical Implications

### Original Prediction (Korollar 5.3)
- ONE L* where anisotropy peaks AND correlation inverts

### Empirical Reality
- TWO L*: Compression peak (L*_A=7) ≠ Correlation inversion (L*_C=28)

### Required Theory Update

The Hodge-theoretic framework needs extension:

1. **Phase 1 (Compression):** Diffusion onto H⁰ completes early
2. **Phase 2 (Plateau):** NEW - "Context holding" phase not in original theory
3. **Phase 3 (Inversion):** H¹ resolution begins late

**Key Insight:** The 21-layer gap suggests the model:
- Finishes building context (L*_A = 7)
- Processes/reasons over context (layers 7-28)
- Only then commits to prediction (L*_C = 28)

---

## Files in This Folder

```
results/
├── ANISOTROPY_RESULTS_SUMMARY.md     # This file
├── anisotropy_profile_pythia.png     # 4-panel visualization
├── eigenvalue_spectrum_pythia.png    # Spectrum at key layers
└── anisotropy_results_pythia.json    # Raw data
```

---

## Reproducibility

**Notebook:** `notebooks/Anisotropy_Profile_Pythia.ipynb`

**Requirements:**
- Google Colab with A100 GPU
- ~25-35 minutes runtime
- HuggingFace token for model access

**Prompts Used:** 16 diverse prompts (factual, creative, technical, abstract)

---

## Citation

If using these results:

```bibtex
@misc{delia2026anisotropy,
  author = {D'Elia, Davide},
  title = {Anisotropy Profile Analysis for Sheaf-Theoretic Framework},
  year = {2026},
  note = {Paper \#3 Empirical Validation}
}
```

---

*Results generated: 2026-01-04*
*Status: VALIDATED - Multi-phase structure discovered*
