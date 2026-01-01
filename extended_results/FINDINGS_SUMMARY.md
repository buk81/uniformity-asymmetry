# Output Correlation Analysis: Key Findings

**Date:** 2026-01-01
**Status:** Preliminary Results (post-paper validation)
**Author:** Davide D'Elia

---

## TL;DR

We ran output-level validation experiments on the Uniformity Asymmetry metric. Four major discoveries:

1. **Gemma's "perfect" correlation (r=0.95) is an artifact** - Remove one category and it flips to r=-0.61
2. **RLHF creates "Deceptive Alignment"** - Llama appears neutral in embeddings but outputs remain biased
3. **Multilingual models show structural inversion** - Apertus compresses concepts differently due to language diversity
4. **Lie vs Bullshit classification** - Categories cluster into "confident wrong" (LIE) vs "uncertain" (BULLSHIT) regimes

---

## Finding 1: The Numeric Artifact

### What we expected
Gemma-2-9B showed r=0.953 correlation between embedding asymmetry (UA) and output preference. This seemed to validate the metric.

### What we found
**This correlation is driven entirely by one category: Ground Truth Numeric.**

| Model | Full Correlation | Without GT-Numeric | Change |
|-------|-----------------|-------------------|--------|
| **Gemma** | r = +0.95*** | r = **-0.61** | Collapses |
| Llama | r = +0.52 | r = -0.49 | Inverts |
| Mistral | r = +0.77 | r = -0.34 | Inverts |
| **Apertus** | r = -0.07 | r = **+0.59** | Emerges |

### Interpretation
Numeric facts (like "Pi = 3.14159") activate a separate processing pathway. All models strongly prefer outputting the concrete number regardless of embedding structure.

**See:** `figures/fig1_correlation_collapse.png`

---

## Finding 2: RLHF Masking (Scientific Facts Category)

### The Problem
If RLHF successfully removes bias, both embeddings AND outputs should be neutral.

### What we found
In the **Scientific Facts** category, Llama-3.1-8B shows **neutral embeddings but biased outputs**.

| Model | Embedding UA | Output Preference | Interpretation |
|-------|--------------|-------------------|----------------|
| Llama | -0.01 (neutral) | +1.17 (80% Side A) | **MASKED** |
| Apertus | +0.11 (biased) | +1.28 (80% Side A) | Aligned |

> **Note:** Apertus is "Aligned" here (Scientific Facts) but "Inverted" in Finding 3 (Numeric Facts). This category-specific behavior is a key insight.

### Interpretation
RLHF may create **"Deceptive Alignment"** - making the model **appear** unbiased at the representation level while preserving output-level preferences. This is a form of surface-level compliance without genuine value alignment.

**Implication for AI Safety:** Embedding-based bias detection may miss RLHF-masked biases. Models can "pass" representation-level audits while maintaining biased behavior.

**See:** `figures/fig2_rlhf_masking.png`

---

## Finding 3: Apertus Inversion (Numeric Facts Category)

### The Paradox
Apertus-8B (23 languages) shows the **opposite** pattern from other models on **Numeric Facts** (GT-Numeric category):
- Other models: Negative UA (prefer concrete numbers in embeddings)
- Apertus: **Positive UA** (prefer abstract concepts in embeddings)

### Explanation: Representational Necessity
Multilingual models must store concepts efficiently across languages. "Pi" exists in all languages, but "3.14159..." is universal. Apertus may cluster the **concept** tightly while the output layer fires the **number**.

| Aspect | Gemma | Apertus |
|--------|-------|---------|
| Embedding | Prefers number | Prefers concept |
| Output | 93% number | 90% number |
| Alignment | Yes | **No (inverted)** |

**See:** `figures/fig3_contribution_heatmap.png`

---

## Finding 4: Lie vs Bullshit Framework

### From Archive Research
We applied a classification framework based on model confidence and asymmetry:

| Regime | Characteristics | Interpretation |
|--------|-----------------|----------------|
| **LIE** | Low NLL + High \|UA\| | Model is **confident** and **asymmetric** |
| **BULLSHIT** | High NLL + Low \|UA\| | Model is **uncertain** and **symmetric** |
| **MIXED** | Other combinations | No clear classification |

### Classification Results

| Classification | Count | Examples |
|----------------|-------|----------|
| LIE_REGIME | 5 | GT-Numeric (Llama, Mistral, Apertus) |
| BULLSHIT_REGIME | 6 | Business, Lifestyle (Gemma) |
| MIXED | 13 | Remaining categories |

### Interpretation
- **GT-Numeric falls into LIE regime**: Models are confident the number is correct
- **Business/Lifestyle fall into BULLSHIT regime**: Models are uncertain about opinion-based framing

**See:** `figures/fig4_category_quadrants.png`

---

## Data Files

| File | Contents |
|------|----------|
| `data/correlation_analysis.json` | Full statistical results |
| `data/contribution_by_category.csv` | Which categories drive correlation |
| `data/cross_model_divergence.csv` | Model disagreement analysis |
| `data/category_classification.csv` | Lie vs Bullshit framework classification |

---

## Provenance

These findings are timestamped on the Bitcoin blockchain:

```
SHA256: 5520612f4ba370c098caf954f9a03a1f324cdb39ffe102461acfd57a48ad7bfb
Timestamp: 2026-01-01 17:00 UTC
Proof: provenance/v2_findings_20260101.tar.gz.ots
```

---

## Implications for the Paper

The original paper's cautious framing ("exploratory metric") was **correct**. These findings:
- **Do not contradict** the paper (no false claims were made)
- **Complement** the paper with output-level validation
- **Explain** the Apertus anomaly noted in the paper

The paper can be published as-is. These extended results provide additional context.

---

*Analysis completed: 2026-01-01*
