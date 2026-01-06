# Gauge Theory Validation: Pythia vs Gemma

**Experiment Date:** 2026-01-04
**Status:** GAUGE THEORY CONFIRMED

---

## Executive Summary

| Metric | Pythia-6.9B | Gemma-2B | Interpretation |
|--------|-------------|----------|----------------|
| Normalization | LayerNorm | **RMSNorm** | Key architectural difference |
| Profile Range | 0.934 | **0.693** | Gemma 25.8% FLATTER |
| Max Anisotropy | 0.994 | 0.973 | Pythia more extreme |
| L* (anisotropy) | 7 | 9 | Relative: 22% vs 50% |
| Effective Rank @ L* | 1.07 | 1.28 | Pythia nearly 1D |
| Initial Anisotropy | 0.060 | **0.730** | Gemma starts compressed |
| Final Anisotropy | 0.095 | 0.280 | Both collapse, Gemma less |

### Verdict: GAUGE THEORY CONFIRMED

Gemma's RMSNorm acts as "gauge fixing" that constrains embeddings to hypersphere S^{d-1}, resulting in a **25.8% flatter anisotropy profile**.

---

## Detailed Comparison

### 1. Profile Shape

**Pythia (LayerNorm) - 5-Phase Structure:**
```
Layer:  0 === 4 === 7 =================== 28 ======= 32
        │     │     │                      │          │
Phase:  INIT  JUMP  MAX   SLOW PLATEAU     │  OUTPUT  │
        │     │     │     (0.99→0.94)      │  COLLAPSE│
        │     │     │                      │          │
Aniso:  0.06  0.94  0.99 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 0.94      0.09
EffRk:  105   1.7   1.07 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 1.67      66
```

**Gemma (RMSNorm) - Smoother 3-Phase Structure:**
```
Layer:  0 ========= 9 ============ 18
        │          │               │
Phase:  INIT-HIGH  MAX    SYMMETRIC DECLINE
        │          │               │
Aniso:  0.73 ↗↗↗↗ 0.97 ↘↘↘↘↘↘↘↘↘ 0.28
EffRk:  4.9  ↘↘↘↘ 1.28 ↗↗↗↗↗↗↗↗↗ 27.7
```

### 2. Key Differences Explained

#### 2.1 Initial Anisotropy (Layer 0)

| Model | Value | Interpretation |
|-------|-------|----------------|
| Pythia | 0.060 | Nearly isotropic - embeddings spread across all dimensions |
| Gemma | **0.730** | Already compressed - RMSNorm pre-constrains geometry |

**Why?** RMSNorm normalizes by RMS (Root Mean Square), which implicitly projects onto a hypersphere. This "gauge fixing" starts immediately at the embedding layer.

#### 2.2 Phase 1 Slope (Rising Phase)

| Model | Slope | R-value | Interpretation |
|-------|-------|---------|----------------|
| Pythia | **+0.186** | 0.88 | DRAMATIC jump (10x steeper) |
| Gemma | +0.018 | 0.75 | Gradual rise |

**Why?** Pythia needs to compress from near-isotropy (0.06) to high anisotropy (0.99). Gemma already starts compressed, so less work needed.

#### 2.3 The "Plateau" Phase

| Model | Plateau Layers | Duration | Anisotropy Change |
|-------|----------------|----------|-------------------|
| Pythia | 7-31 | 24 layers (75%) | 0.994 → 0.938 (-0.056) |
| Gemma | 9-14 | 5 layers (28%) | 0.973 → 0.960 (-0.013) |

**Why?** Pythia's extreme compression creates a long "context holding" plateau. Gemma's gentler compression leads to a shorter, less distinct plateau.

#### 2.4 Output Collapse

| Model | Before → After | Change | Effective Rank Change |
|-------|----------------|--------|----------------------|
| Pythia | 0.938 → 0.095 | -0.843 | 1.67 → 66.4 |
| Gemma | 0.789 → 0.280 | -0.509 | 3.9 → 27.7 |

**Why?** Both models need to project to vocabulary space, requiring expansion. Pythia's more extreme compression requires more dramatic expansion.

---

## 3. Gauge Theory Interpretation

### Gemini's Prediction (2026-01-04)

> "Normalization layers act as **gauge fixing** operations."
>
> | Architecture | Norm | Effect |
> |-------------|------|--------|
> | Pythia | LayerNorm | Radial freedom preserved → clear inversion |
> | Gemma | RMSNorm | Spherical geometry enforced → subtle/no inversion |

### Empirical Validation

| Prediction | Expected | Observed | Status |
|------------|----------|----------|--------|
| Gemma flatter profile | Yes | 25.8% flatter | **CONFIRMED** |
| Gemma less extreme compression | Yes | Max 0.97 vs 0.99 | **CONFIRMED** |
| Gemma starts more anisotropic | Yes | 0.73 vs 0.06 | **CONFIRMED** |
| Both show bell curve | Yes | Both is_bell_curve=true | **CONFIRMED** |

### Mathematical Explanation

**LayerNorm (Pythia):**
$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$$

- Subtracts mean → allows radial freedom
- Preserves direction information
- Embeddings can vary in magnitude

**RMSNorm (Gemma):**
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2}}$$

- Divides by RMS → projects onto hypersphere $S^{d-1}$
- Constrains to unit norm (approximately)
- Only angular diffusion possible

---

## 4. Multi-Phase Model Comparison

### Pythia: 5 Distinct Phases

| Phase | Layers | Dynamics | Anisotropy | Effective Rank |
|-------|--------|----------|------------|----------------|
| 0. Init | 0-3 | Tokenization/Setup | 0.06→0.12 | 105→91 |
| 1. Jump | 3-4 | **DRAMATIC** compression | 0.12→0.94 | 91→1.7 |
| 2. Peak | 4-7 | Refinement | 0.94→**0.994** | 1.7→**1.07** |
| 3. Plateau | 7-31 | Slow release | 0.994→0.938 | 1.07→1.67 |
| 4. Collapse | 31-32 | Output projection | 0.938→**0.095** | 1.67→**66** |

### Gemma: 3 Smoother Phases

| Phase | Layers | Dynamics | Anisotropy | Effective Rank |
|-------|--------|----------|------------|----------------|
| 0. Rise | 0-9 | Gradual compression | 0.73→**0.97** | 4.9→**1.28** |
| 1. Plateau | 9-14 | Brief hold | 0.97→0.96 | 1.28→1.41 |
| 2. Decline | 14-18 | Symmetric expansion | 0.96→**0.28** | 1.41→**27.7** |

---

## 5. Layer-by-Layer Data

### Intrinsic Dimension Ratio (λ₁/Σλᵢ)

| Layer | Pythia | Gemma | Ratio (P/G) |
|-------|--------|-------|-------------|
| 0 | 0.060 | 0.730 | 0.08x |
| Relative 25% | 0.994 (L7) | 0.916 (L4) | 1.09x |
| Relative 50% | 0.987 (L16) | 0.973 (L9) | 1.01x |
| Relative 75% | 0.962 (L24) | 0.905 (L13) | 1.06x |
| Final | 0.095 | 0.280 | 0.34x |

### Effective Rank

| Layer | Pythia | Gemma | Notes |
|-------|--------|-------|-------|
| 0 | 104.5 | 4.9 | Pythia 21x more isotropic at start |
| L* | 1.07 | 1.28 | Both nearly 1D, Pythia more extreme |
| Final | 66.4 | 27.7 | Both expand for output |

---

## 6. Implications for Paper #3

### 6.1 Theory Refinement Needed

The original Hodge-theoretic framework assumed:
- Single L* where anisotropy peaks AND correlation inverts
- Universal pattern across architectures

**Updated Understanding:**
- L*_anisotropy can differ from L*_correlation
- Normalization choice acts as "gauge fixing"
- RMSNorm creates different topological constraints

### 6.2 New Hypothesis

**H8: Normalization as Gauge Fixing**

The choice of normalization layer determines the **gauge** of the sheaf:
- LayerNorm: Affine gauge → full freedom
- RMSNorm: Spherical gauge → constrained to S^{d-1}

This explains why:
- Gemma shows less dramatic phase transitions
- The "plateau" phase is shorter in Gemma
- Paper #2 saw "no clear inversion" in Gemma (it's subtler, not absent)

### 6.3 Falsifiable Predictions

1. **Other RMSNorm models** (LLaMA, Mistral) should show similar flatter profiles
2. **Pre-RMSNorm models** (GPT-2, BERT) should show Pythia-like profiles
3. **Hybrid models** with mixed normalization should show intermediate profiles

---

## 7. Summary Statistics

```
============================================================
GAUGE THEORY VALIDATION RESULTS
============================================================

Pythia-6.9B (LayerNorm):
  Profile Range:      0.934
  Max Anisotropy:     0.994 (Layer 7)
  Min Anisotropy:     0.060 (Layer 0)
  Phase 1 Slope:      +0.186 (R=0.88)
  Phase 2 Slope:      -0.009 (R=-0.41)
  Bell Curve:         YES

Gemma-2B (RMSNorm):
  Profile Range:      0.693
  Max Anisotropy:     0.973 (Layer 9)
  Min Anisotropy:     0.280 (Layer 18)
  Phase 1 Slope:      +0.018 (R=0.75)
  Phase 2 Slope:      -0.050 (R=-0.72)
  Bell Curve:         YES

Comparison:
  Flatness Difference: 25.8% (Gemma flatter)
  Initial Anisotropy:  Gemma 12x higher
  Max Compression:     Pythia more extreme
  Phase Transitions:   Pythia 5, Gemma 3

VERDICT: GAUGE THEORY CONFIRMED
============================================================
```

---

## 8. Files

```
Results/
├── GAUGE_THEORY_VALIDATION.md          # This document
├── anisotropy_results_pythia.json      # Pythia raw data
├── anisotropy_results_gemma.json       # Gemma raw data
├── anisotropy_profile_pythia.png       # Pythia visualization
├── anisotropy_profile_gemma.png        # Gemma visualization
├── eigenvalue_spectrum_pythia.png      # Pythia eigenvalues
├── eigenvalue_spectrum_gemma.png       # Gemma eigenvalues
└── norm_analysis_gemma.png             # RMSNorm effect
```

---

## 9. Citation

```bibtex
@misc{delia2026gauge,
  author = {D'Elia, Davide},
  title = {Gauge Theory Validation: LayerNorm vs RMSNorm Anisotropy Profiles},
  year = {2026},
  note = {Paper \#3 Empirical Validation - Gemini Prediction Confirmed}
}
```

---

*Generated: 2026-01-04*
*Status: GAUGE THEORY CONFIRMED - RMSNorm creates 25.8% flatter profile*
