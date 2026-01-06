# Pythia Anomaly Investigation

**Date:** 2026-01-05
**Status:** IN PROGRESS
**Question:** Why does Pythia show dampening (Gain = 0.80) while GPT-2 shows expansion (Gain = 1.01-1.16)?

---

## The Anomaly

Both Pythia and GPT-2 use **LayerNorm**, but show opposite residual stream dynamics:

| Model | Norm Type | Residual Stream Gain | Behavior |
|-------|-----------|---------------------|----------|
| Pythia-6.9B | LayerNorm | **0.80** | DAMPENING |
| GPT2-XL | LayerNorm | **1.02** | EXPANSION |
| GPT2-Large | LayerNorm | **1.01** | EXPANSION |
| GPT2-Medium | LayerNorm | **1.16** | EXPANSION |

**Conclusion:** LayerNorm is NOT the cause of dampening.

---

## Architectural Differences

### 1. PARALLEL vs SEQUENTIAL Attention (Primary Suspect)

**GPT-2 (Sequential):**
```
h = x + Attention(LayerNorm(x))
output = h + MLP(LayerNorm(h))
```

**Pythia/GPT-NeoX (Parallel):**
```
output = x + Attention(LayerNorm(x)) + MLP(LayerNorm(x))
```

**Key Difference:**
- GPT-2: MLP sees the OUTPUT of Attention (already transformed)
- Pythia: MLP sees the SAME input as Attention (parallel computation)

**Hypothesis:** Parallel architecture causes both Attention and MLP to "fight" over the same input, leading to net contraction when combined.

### 2. Position Embeddings

| Model | Position Encoding |
|-------|------------------|
| GPT-2 | Learned absolute positions |
| Pythia | Rotary Position Embeddings (RoPE) |

**Hypothesis:** RoPE's rotational structure may interact differently with the residual stream dynamics.

### 3. Training Data

| Model | Training Data | Size |
|-------|--------------|------|
| GPT-2 | WebText (curated web pages) | ~40GB |
| Pythia | The Pile (diverse sources) | 800GB |

**Hypothesis:** The Pile's diversity may have trained Pythia to be more "conservative" (dampening).

### 4. Normalization Placement

Both use LayerNorm, but the PLACEMENT differs in parallel vs sequential:
- GPT-2: Two separate LayerNorms applied sequentially
- Pythia: Single LayerNorm applied to input, then split to Attention AND MLP

---

## Evidence from Scaling Law Data

From `scaling_law_multi_pythia_results.json`:

### All 8 Pythia Models (70M - 12B)

| Model | Attn Contract % | Last MLP Gain | Architecture |
|-------|-----------------|---------------|--------------|
| pythia-70m | 100% | 1.50 | GPT-NeoX (Parallel) |
| pythia-160m | 100% | 2.82 | GPT-NeoX (Parallel) |
| pythia-410m | 100% | 1.78 | GPT-NeoX (Parallel) |
| pythia-1b | 100% | 3.72 | GPT-NeoX (Parallel) |
| pythia-1.4b | 100% | 3.52 | GPT-NeoX (Parallel) |
| pythia-2.8b | 100% | 2.10 | GPT-NeoX (Parallel) |
| pythia-6.9b | 97% | 6.30 | GPT-NeoX (Parallel) |
| pythia-12b | 97% | 7.71 | GPT-NeoX (Parallel) |

**Universal Pattern:**
- ✅ Attention ALWAYS contracts (97-100%)
- ✅ Last MLP Gain is ALWAYS > 1.0

But RESIDUAL STREAM for Pythia-6.9B = 0.80 (CONTRACTS!)

**This means:** The parallel combination of contracting Attention + expanding MLP results in NET CONTRACTION in the residual stream.

---

## The Parallel Architecture Hypothesis

### Mathematical Model

**GPT-2 (Sequential):**
```
h = x + A(x)           // Attention adds to residual
out = h + M(h)         // MLP adds to already-transformed h
out = x + A(x) + M(x + A(x))
```

**Pythia (Parallel):**
```
out = x + A(x) + M(x)  // Both add to SAME original x
```

### Why Parallel Might Dampen

In the parallel case:
- A(x) and M(x) both operate on the SAME x
- Their outputs are summed directly
- If A(x) contracts and M(x) expands, they may partially CANCEL

In the sequential case:
- M operates on (x + A(x)), a TRANSFORMED input
- The expansion happens on top of the attention output
- Less cancellation, more net expansion

---

## Testable Predictions

### Prediction 1: Other GPT-NeoX Models Should Dampen

If the parallel architecture causes dampening, then:
- **OPT** (uses parallel attention) → Should dampen
- **BLOOM** (uses parallel attention) → Should dampen
- **Standard LLaMA** (sequential) → Should NOT dampen (already confirmed: 1.48)

### Prediction 2: GPT-J Should Dampen (GPT-NeoX based)

GPT-J-6B uses GPT-NeoX architecture → Should show Gain < 1.0

### Prediction 3: The "Cancellation" Effect

If we measure A(x) and M(x) separately and add them:
- |A(x)| + |M(x)| should be LARGER than |A(x) + M(x)|
- This would prove partial cancellation

---

## Next Steps

1. **Test GPT-J-6B** - Another GPT-NeoX model
2. **Test OPT-6.7B** - Different parallel architecture
3. **Measure A(x) and M(x) separately** in Pythia
4. **Compare vector directions** - Are Attention and MLP outputs opposing?

---

## Current Conclusion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LEADING HYPOTHESIS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   The PARALLEL ATTENTION architecture (GPT-NeoX) causes dampening.          │
│                                                                              │
│   Mechanism:                                                                 │
│   ├── Both Attention and MLP operate on the SAME input                     │
│   ├── Their outputs partially CANCEL when added                            │
│   └── Net result: Residual stream contracts                                │
│                                                                              │
│   Evidence:                                                                  │
│   ├── ALL Pythia models (8 sizes) show same pattern                        │
│   ├── GPT-2 (sequential) shows opposite behavior                           │
│   ├── LayerNorm is present in BOTH → not the cause                         │
│                                                                              │
│   To Confirm: Test GPT-J-6B (also GPT-NeoX)                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## GPT-J-6B Validation Test

**Notebook:** `notebooks/GPTJ_Parallel_Architecture_Test.ipynb`

**Test Subject:**
- Model: GPT-J-6B (EleutherAI)
- Architecture: GPT-NeoX (Parallel Attention)
- Parameters: 6B
- Normalization: LayerNorm

**Expected Result (if hypothesis correct):**
- GPT-J-6B should show Gain < 1.0 (dampening)
- Similar to Pythia-6.9B (0.80)
- Opposite to GPT-2 (1.01-1.16)

**Decision Tree:**
```
IF GPT-J-6B Gain < 1.0:
    → PARALLEL ARCHITECTURE HYPOTHESIS: CONFIRMED
    → Both GPT-NeoX models (Pythia, GPT-J) dampen
    → Architecture determines base level, not LayerNorm

IF GPT-J-6B Gain > 1.0:
    → PARALLEL ARCHITECTURE HYPOTHESIS: REJECTED
    → Pythia's dampening is NOT explained by architecture
    → Need to investigate: training data, RoPE, other factors
```

---

*Investigation started: 2026-01-05*
*Status: HYPOTHESIS FORMED - GPT-J test notebook ready*
*Next: Run GPTJ_Parallel_Architecture_Test.ipynb on Colab*
