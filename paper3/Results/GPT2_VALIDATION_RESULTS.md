# GPT-2 LayerNorm Validation Results

**Date:** 2026-01-05 15:03
**Status:** HYPOTHESIS REJECTED
**Purpose:** Test if LayerNorm universally causes dampening (Gain < 1.0)

---

## Executive Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CRITICAL FINDING                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   LAYERNORM DAMPENING HYPOTHESIS:  ❌ REJECTED                              │
│                                                                              │
│   GPT-2 models (all LayerNorm) show Gain > 1.0 (EXPANSION)                  │
│   This contradicts the prediction based on Pythia-6.9B (0.80)               │
│                                                                              │
│   Conclusion: Pythia's dampening is PYTHIA-SPECIFIC,                        │
│               NOT a LayerNorm property!                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Experiment Design

### Models Tested

| Model | Parameters | Norm Type | Layers | Role |
|-------|------------|-----------|--------|------|
| GPT2-XL | 1.5B | LayerNorm | 48 | Primary validation |
| GPT2-Large | 774M | LayerNorm | 36 | Secondary validation |
| GPT2-Medium | 355M | LayerNorm | 24 | Tertiary validation |

### Reference Models (from Grand Unified Benchmark)

| Model | Gain | Norm Type | Prediction for GPT-2 |
|-------|------|-----------|---------------------|
| Pythia-6.9B | 0.80 | LayerNorm | < 1.0 (Dampening) |
| Mistral-7B | 1.11 | RMSNorm | > 1.0 (Expansion) |
| LLaMA-3.1-8B | 1.48 | RMSNorm | > 1.0 (Expansion) |
| Gemma-7B | 2.31 | RMSNorm | > 1.0 (Expansion) |

### Prompt Set

- 25 prompts across 5 categories (same as Grand Unified Benchmark)
- Categories: Factual, Syntactic, Cliche, Novel, Nonsense
- Total measurements: 75 (3 models × 25 prompts)

---

## 2. Results

### 2.1 Base Level (Mean Gain)

| Model | Mean Gain | Std | Min | Max | Status |
|-------|-----------|-----|-----|-----|--------|
| **GPT2-XL** | **1.019** | 0.025 | 0.981 | 1.073 | **> 1.0** |
| **GPT2-Large** | **1.014** | 0.025 | 0.981 | 1.070 | **> 1.0** |
| **GPT2-Medium** | **1.163** | 0.077 | 1.065 | 1.398 | **> 1.0** |

### 2.2 Statistical Test (t-test vs 1.0)

| Model | t-statistic | p-value (one-sided) | Is_Dampening | Status |
|-------|-------------|---------------------|--------------|--------|
| GPT2-XL | +3.84 | 0.9996 | **FALSE** | **REJECTED** |
| GPT2-Large | +2.84 | 0.9955 | **FALSE** | **REJECTED** |
| GPT2-Medium | +10.57 | ~1.0 | **FALSE** | **REJECTED** |

**All three models show statistically significant EXPANSION (Gain > 1.0), not dampening.**

### 2.3 Bentov Law Correlation (|Gain - 1.0| vs Entropy)

| Model | Correlation r | p-value | Direction | Pythia Prediction |
|-------|---------------|---------|-----------|-------------------|
| GPT2-XL | +0.045 | 0.832 | Positive | Negative |
| GPT2-Large | +0.151 | 0.470 | Positive | Negative |
| GPT2-Medium | +0.371 | 0.068 | Positive | Negative |
| Pythia-6.9B (ref) | **-0.199** | — | **Negative** | — |

**GPT-2 shows POSITIVE correlation (like RMSNorm models), not negative like Pythia.**

---

## 3. Comparison: All LayerNorm Models

```
LAYERNORM MODELS GAIN COMPARISON:

Model           Gain    Expected    Actual      Discrepancy
────────────────────────────────────────────────────────────
Pythia-6.9B     0.80    < 1.0       < 1.0       ✓ Match
GPT2-XL         1.02    < 1.0       > 1.0       ✗ OPPOSITE
GPT2-Large      1.01    < 1.0       > 1.0       ✗ OPPOSITE
GPT2-Medium     1.16    < 1.0       > 1.0       ✗ OPPOSITE

Conclusion: 3/4 LayerNorm models show EXPANSION, not dampening.
            Pythia is the OUTLIER, not the rule.
```

---

## 4. Revised Model Hierarchy (7 Models)

```
COMPLETE GAIN HIERARCHY (All Tested Models):

    ←── CONTRACTION ──│── EXPANSION ──→
                      │
    0.80              │  1.01  1.02  1.11  1.16  1.48        2.31
     ▼                │   ▼     ▼     ▼     ▼     ▼           ▼
────●─────────────────┼───●─────●─────●─────●─────●───────────●────
    │                 │   │     │     │     │     │           │
 Pythia            NEUTRAL GPT2  GPT2 Mistr GPT2  LLaMA    Gemma
                   (1.0) Large  XL   -7B  Med   3.1-8B    -7B

LayerNorm models: Pythia(0.80), GPT2-Large(1.01), GPT2-XL(1.02), GPT2-Medium(1.16)
RMSNorm models:   Mistral(1.11), LLaMA(1.48), Gemma(2.31)

OBSERVATION: LayerNorm models SPAN the neutral point!
             They do NOT cluster below 1.0.
```

---

## 5. Theoretical Implications

### 5.1 What This REJECTS

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| "LayerNorm = Dampening" | ❌ REJECTED | GPT-2 shows expansion |
| "RMSNorm = Expansion" | ⚠️ PARTIAL | True for tested RMSNorm models |
| "Normalization determines base level" | ❌ REJECTED | Both norm types span wide range |
| "Pythia represents LayerNorm behavior" | ❌ REJECTED | GPT-2 contradicts Pythia |

### 5.2 What This CONFIRMS

| Finding | Status | Evidence |
|---------|--------|----------|
| "Base level is architecture-dependent" | ✅ CONFIRMED | Different models have different base levels |
| "Input modulates gain" | ✅ CONFIRMED | Nonsense → highest gain (all models) |
| "Plattitüden-Tunnel" | ✅ CONFIRMED | Cliche → lowest entropy (all models) |

### 5.3 Open Questions

1. **Why is Pythia different?**
   - Training data (The Pile vs WebText)?
   - Rotary Position Embeddings?
   - GPT-NeoX architecture specifics?
   - Model size effects?

2. **What determines base level if not normalization?**
   - Attention mechanism details?
   - Training procedure?
   - Tokenizer effects?

---

## 6. Impact on Paper #3

### Must Be Revised

```
OLD CLAIM (WRONG):
"LayerNorm models exhibit dampening (Gain < 1.0),
while RMSNorm models exhibit expansion (Gain > 1.0)."

NEW CLAIM (CORRECT):
"Base level gain varies widely across architectures
(0.80 to 2.31) and is NOT determined by normalization type alone.
Pythia-6.9B is an outlier among LayerNorm models."
```

### Paper #3 Discussion Must Address

1. The failed prediction (scientific honesty)
2. Alternative hypotheses for Pythia's behavior
3. What actually determines base level
4. The value of falsification in science

---

## 7. Files Generated

```
Results/
├── gpt2_layernorm_validation_20260105_150306.csv   # Raw data
├── gpt2_layernorm_validation_20260105_150306.json  # Full results
├── gpt2_layernorm_validation_20260105_150306.png   # Main visualization
├── gpt2_vs_reference_20260105_150306.png           # Comparison chart
└── GPT2_VALIDATION_RESULTS.md                      # This document
```

---

## 8. Critical Note on Gemini's Misinterpretation

**WARNING:** Gemini 2.0 Flash incorrectly interpreted these results, claiming:
- "GPT-2 Average Gain: ~0.30x" (WRONG - actual: 1.01-1.16)
- "LayerNorm dämpft IMMER" (WRONG - GPT-2 shows expansion)
- "Hypothese bestätigt" (WRONG - hypothesis was REJECTED)

**Possible cause:** Confusion between `Last_Gain` (what we measure, ~1.0) and `Total_Amp` (cumulative amplification, ~15-25x).

**The JSON file explicitly states:**
```json
"hypothesis_status": {
    "layernorm_dampening": "REJECTED",
    "bentov_inversion": "REJECTED"
}
```

---

## 9. Conclusion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FINAL VERDICT                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   The GPT-2 validation test has FALSIFIED the LayerNorm dampening           │
│   hypothesis. This is a valuable scientific result.                          │
│                                                                              │
│   Key Findings:                                                              │
│   1. LayerNorm does NOT universally cause dampening                         │
│   2. Pythia-6.9B is an outlier, not representative of LayerNorm             │
│   3. Base level is determined by factors OTHER than normalization type      │
│   4. The Bentov Law inversion is also Pythia-specific                       │
│                                                                              │
│   Implication for Paper #3:                                                  │
│   The paper must discuss this FALSIFICATION honestly and explore            │
│   alternative hypotheses for what determines base level gain.               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Generated: 2026-01-05 15:30*
*Data source: gpt2_layernorm_validation_20260105_150306.json*
*Status: HYPOTHESIS REJECTED - Documentation complete*
