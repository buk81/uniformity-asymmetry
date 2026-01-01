# Output-Correlation v2 Phase 1: Full Analysis

*Date: 2026-01-01*
*Status: Paradigm Shift - Revised Interpretation*

---

## Executive Summary

**Phase 1 has FUNDAMENTALLY revised the v1 interpretation.**

| v1 Interpretation | v2 Revised Interpretation |
|-------------------|---------------------------|
| Gemma shows perfect UA→Output coupling | GT-Numeric ALONE drives Gemma's r=0.953 |
| Apertus shows no correlation | Apertus shows r=0.587 WITHOUT GT-Numeric |
| UA validated as output predictor | UA works ONLY for non-numeric facts |

**The central discovery:** Numeric facts activate a separate processing pathway that pulls all models toward the concrete number (Side B) - regardless of embedding structure.

---

## 1. The GT-Numeric Effect

### 1.1 Correlation With vs Without GT-Numeric

| Model | r (Full) | r (w/o GT-Num) | Δr | Interpretation |
|-------|----------|----------------|-----|----------------|
| **Gemma** | +0.953*** | **-0.608** | -1.56 | Correlation COLLAPSES |
| Llama | +0.518 | -0.486 | -1.00 | Correlation INVERTS |
| Mistral | +0.769* | -0.341 | -1.11 | Correlation INVERTS |
| **Apertus** | -0.070 | **+0.587** | +0.66 | Correlation EMERGES |

### 1.2 Contribution Analysis

```
                    Gemma    Llama   Mistral  Apertus
─────────────────────────────────────────────────────
GT-Numeric          0.914    0.045    0.180   -0.168  ← DOMINATES
GT-NonNumeric       0.005    0.004    0.018    0.033
Scientific          0.024   -0.015    0.010    0.139
Lifestyle           0.002    0.000   -0.002    0.005
Business            0.008    0.006    0.002    0.002
Tech-Philosophy     0.007   -0.007   -0.023    0.024
─────────────────────────────────────────────────────
TOTAL               0.960    0.033    0.185    0.035
```

**Key insight:** Gemma's contribution comes **95%** from GT-Numeric (0.914 / 0.960).

### 1.3 Visualization

```
         Gemma r=0.953              Without GT-Numeric
              │                          │
    GT-Num ●──┼─────── r=0.95           ┼────── r=-0.61
              │    ↗                     │
         ●●●●●●                     ●●●●●● (scattered)
              │                          │
              └────────────              └────────────

The SINGLE point (GT-Numeric) creates the entire correlation!
```

---

## 2. The Apertus Paradox Resolved

### 2.1 Original Observation (v1)

```
Apertus: r = -0.07, p = 0.89 → "No correlation"
```

### 2.2 New Interpretation (v2)

```
Apertus WITHOUT GT-Numeric: r = +0.587 → "Moderate positive correlation!"

The GT-Numeric inversion (-0.168 contribution) DESTROYS
the otherwise positive correlation.
```

### 2.3 Apertus vs Gemma on GT-Numeric

| Metric | Gemma | Apertus | Interpretation |
|--------|-------|---------|----------------|
| UA | -0.251 | **+0.097** | Gemma aligned, Apertus inverted |
| Output Pref | -3.64 | -1.73 | Both → Side B (number) |
| % Side B | 93% | 90% | Both prefer the number |
| Contribution | +0.914 | **-0.168** | Gemma aligned, Apertus misaligned |

**The Inversion:**
- Gemma: Embedding AND output prefer Side B → aligned
- Apertus: Embedding prefers Side A, output prefers Side B → **INVERSION**

### 2.4 Why Does Only Apertus Invert?

**Hypothesis: Representational Necessity**

```
Apertus (multilingual, 23 languages):
├── Must store "π" as concept (same for all languages)
├── → Embedding compresses to "Archimedes Constant" (Side A)
├── But output must deliver concrete number
└── → Output fires "3.14159..." (Side B)

Gemma (primarily English):
├── No necessity for concept compression
├── → Embedding and output both on "3.14159..."
└── → No conflict
```

---

## 3. RLHF Masking Confirmed

### 3.1 Scientific Facts Comparison

| Model | UA (Embedding) | Output Pref | Interpretation |
|-------|----------------|-------------|----------------|
| Llama | **-0.013** | +1.17 | **MASKED** - neutral emb, biased out |
| Mistral | -0.012 | +0.97 | **MASKED** |
| Gemma | +0.021 | +1.14 | WEAK |
| Apertus | **+0.109** | +1.28 | **ALIGNED** - biased emb, biased out |

### 3.2 The Masking Pattern

```
                 Embedding          Output
                 ─────────          ──────
Llama:           ≈ neutral          biased (80% A)
Apertus:         biased             biased (80% A)

→ Same output bias, but Llama "hides" it in the embedding!
→ RLHF neutralizes representations, not outputs
```

### 3.3 Implications for AI Safety

```
┌─────────────────────────────────────────────────────────────┐
│  WARNING: Embedding-based bias detection can fail!          │
│                                                             │
│  • Llama appears "neutral" in embeddings                    │
│  • But output is identically biased as Apertus              │
│  • RLHF creates "Deceptive Alignment"                       │
│  • The bias is hidden, not removed                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. YMYL Hypothesis Refuted

### 4.1 Original Hypothesis

> "Gemma's high correlation is driven by Topic-Specific Training (YMYL)."

### 4.2 Result

| Model | YMYL Contribution | Non-YMYL Contribution | YMYL stronger? |
|-------|-------------------|----------------------|----------------|
| Gemma | +0.013 | +0.233 | ✗ NO |
| Llama | -0.007 | +0.012 | ✗ NO |
| Mistral | +0.004 | +0.044 | ✗ NO |
| Apertus | +0.072 | -0.027 | ✓ YES |

**Conclusion:** YMYL categories do NOT show stronger correlation. The hypothesis is refuted.

---

## 5. Cross-Model Divergence

### 5.1 Divergence Ranking

| Category | UA Std | Pref Std | Total Divergence |
|----------|--------|----------|------------------|
| **ground_truth_numeric** | **0.133** | **1.256** | **1.390** |
| ground_truth_nonnumeric | 0.037 | 0.147 | 0.184 |
| scientific_facts | 0.046 | 0.109 | 0.155 |
| tech_philosophy | 0.025 | 0.108 | 0.132 |
| business | 0.008 | 0.106 | 0.114 |
| lifestyle | 0.020 | 0.083 | 0.103 |

**GT-Numeric has 7.5× higher divergence than the next category!**

### 5.2 Interpretation

```
GT-Numeric is FUNDAMENTALLY different:
├── Highest UA variance (0.133 vs 0.008-0.046)
├── Highest preference variance (1.256 vs 0.083-0.147)
├── Only category with inversion (Apertus)
└── Drives 95% of correlation in Gemma
```

---

## 6. Revised Paper Narrative

### 6.1 OLD (v1)

> "Uniformity Asymmetry correlates almost perfectly with output preference in Gemma (r=0.953). This validates UA as a predictor for output bias."

### 6.2 NEW (v2)

> "The seemingly perfect correlation in Gemma (r=0.953) is a statistical artifact. It is driven exclusively by the GT-Numeric category, which represents a separate processing pathway for numeric facts.
>
> The actual discovery is twofold:
> 1. **RLHF Masking:** Llama shows neutral embedding but biased output - RLHF creates 'Deceptive Alignment' without actual debiasing.
> 2. **Numeric Inversion:** Apertus is the only model showing inversion on numeric facts - the multilingually necessary concept embedding conflicts with the number-preferring output."

---

## 7. New Hypotheses

### H14: The "Numeric Pathway" Hypothesis (confirmed)

> Numeric facts activate a separate processing pathway that prefers the concrete number regardless of embedding structure.

**Evidence:**
- All models: Output prefers Side B (number)
- Apertus: Only model with positive UA for Numeric
- GT-Numeric: 7.5× higher divergence than other categories

### H15: The "Representational Debt" Hypothesis (strengthened)

> Multilingual models accumulate "Representational Debt" - necessary concept compression that leads to embedding-output discrepancies.

**Evidence:**
- Apertus (23 languages): Only Numeric inversion
- Gemma/Llama/Mistral (primarily English): No inversion

### H16: The "Deceptive Alignment" Problem (confirmed)

> RLHF creates superficial neutrality in embeddings without removing output bias. This constitutes a form of deceptive alignment where models appear aligned at the representation level while maintaining misaligned behavior.

**Evidence:**
- Llama: UA ≈ 0, Output = +1.17 (80% A)
- Apertus: UA = +0.11, Output = +1.28 (80% A)
- Identical output bias, different embeddings

---

## 8. Implications

### 8.1 For the Paper

| Claim | Status | Evidence |
|-------|--------|----------|
| UA validated as output predictor | ⚠️ LIMITED | Only for non-numeric categories |
| Gemma shows perfect coupling | ❌ REFUTED | Artifact of GT-Numeric |
| RLHF Masking exists | ✅ CONFIRMED | Llama vs Apertus comparison |
| Numeric Pathway separate | ✅ NEWLY DISCOVERED | 7.5× divergence |

### 8.2 For AI Safety

```
1. Embedding metrics are NOT reliable bias detectors
2. RLHF can "hide" bias without removing it
3. Numeric facts behave fundamentally differently
4. Multilingual models have structural bias risks
```

### 8.3 For v2 Phase 2

1. **Per-Statement Analysis** - Which numeric pairs drive the inversion?
2. **Layer-wise Analysis** - Where do Concept vs Number pathways diverge?
3. **More multilingual models** - Is Apertus pattern typical?

---

## 9. Files

### Analysis Outputs

```
data/
├── correlation_analysis.json      # All results
├── cross_model_divergence.csv     # Divergence per category
├── contribution_by_category.csv   # Contribution per Model×Category
└── category_classification.csv    # Lie vs Bullshit classification

figures/
├── fig1_correlation_collapse.png  # Correlation collapses without GT-Numeric
├── fig2_rlhf_masking.png          # Llama Masking vs Apertus Alignment
├── fig3_contribution_heatmap.png  # GT-Numeric dominates Gemma (0.914)
├── fig4_category_quadrants.png    # Lie vs Bullshit framework
└── fig5_rlhf_divergence_detail.png
```

---

## 10. Summary

### The Three Central Insights

```
┌─────────────────────────────────────────────────────────────┐
│  1. GEMMA's r=0.953 IS AN ARTIFACT                          │
│     → GT-Numeric alone drives 95% of the correlation        │
│     → Without GT-Numeric: r = -0.608 (negative!)            │
├─────────────────────────────────────────────────────────────┤
│  2. APERTUS IS THE "HONEST" MODEL                           │
│     → Shows r = +0.587 for non-numeric categories           │
│     → GT-Numeric inversion is "Representational Debt"       │
├─────────────────────────────────────────────────────────────┤
│  3. RLHF CREATES "SURFACE ALIGNMENT"                        │
│     → Llama: Neutral in embedding, biased in output         │
│     → Bias is hidden, not removed                           │
└─────────────────────────────────────────────────────────────┘
```

### What Remains from the v1 Paper?

| v1 Claim | v2 Status |
|----------|-----------|
| UA is a valid metric | ✅ Remains (with limitations) |
| Gemma shows strong correlation | ❌ Revised (artifact) |
| Apertus shows anomaly | ✅ Remains (now explained) |
| RLHF effect exists | ✅ Remains (strengthened) |

---

## 11. Next Steps

### For the Paper

1. **Reframe:** From "UA validated" to "Numeric Pathway discovered"
2. **Focus:** RLHF Masking as main contribution
3. **Caveat:** Analyze GT-Numeric separately

### For v2 Phase 2

1. Per-statement analysis of GT-Numeric pairs
2. Attention pattern analysis: Concept vs Number
3. Test more multilingual models

---

*v2 Phase 1 completed: 2026-01-01*
*Paradigm shift documented*
*Next step: Phase 2 or Paper revision*
