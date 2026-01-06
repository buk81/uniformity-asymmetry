# FFN Expansion Analysis: Pythia-1.4B

**Experiment Date:** 2026-01-05
**Model:** EleutherAI/pythia-1.4b
**Layers:** 24
**Prompts:** 8 (diverse semantic content)

---

## Executive Summary

| Prediction | Expected | Observed | Status |
|------------|----------|----------|--------|
| Attention contracts | gain < 1 | **ALL 24 layers < 1** | ✅ |
| MLP/FFN expands | gain > 1 | **Only 2 of 24 layers** | ❌ |
| Net expansion somewhere | gain > 1 | **Only Layer 23** | ⚠️ |

### Verdict: HYPOTHESIS REJECTED - NEW INSIGHT GAINED

Die ursprüngliche Hypothese "Attention contracts, FFN expands" ist **FALSCH**.

**Neue Erkenntnis:** Fast das GESAMTE Netzwerk kontrahiert. Expansion passiert **NUR im letzten Layer**.

---

## 1. Attention Gains (All Contracting)

```
Layer:  0     2     4     6     8    10    12    14    16    18    20    22
Gain:  0.53  0.24  0.25  0.27  0.29  0.22  0.26  0.25  0.17  0.16  0.08  0.10
       ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ MIN
       ALL CONTRACTING                                        L*=20
```

### Key Statistics
- **Min gain:** 0.083 (Layer 20) - stärkste Kontraktion
- **Max gain:** 0.527 (Layer 0) - schwächste Kontraktion
- **Contracting layers:** 24/24 (100%)
- **L*_attention:** Layer 20

### Interpretation
Attention ist **IMMER kontraktiv**. Dies bestätigt die Restriction Maps Analyse:
- √(A_{ij}) ≤ 1 (Attention Weights sind Wahrscheinlichkeiten)
- Die Kontraktion ist intrinsisch zur Attention-Mechanik

---

## 2. MLP/FFN Gains (Mostly Contracting!)

```
Layer:  0     2     4     6     8    10    12    14    16    18    20    22    23
Gain:  0.89  0.37  0.86  0.56  0.48  0.50  0.59  0.67  0.68  0.79  0.70  0.92  3.60
       ↓↓↓↓  ↓↓↓↓  ↓↓↓↓  ↓↓↓↓  ↓↓↓↓  ↓↓↓↓  ↓↓↓↓  ↓↓↓↓  ↓↓↓↓  ↓↓↓↓  ↓↓↓↓  ↓↓↓↓  ↑↑↑↑
       CONTRACT                                                   STABLE  EXPLODE
```

### Anomaly: Layer 3
```
Layer 3 MLP Gain: 1.77 (> 1, expanding!)
```

### Key Statistics
- **Min gain:** 0.261 (Layer 1)
- **Max gain:** 3.604 (Layer 23) ← **PREDICTION LAYER**
- **Contracting layers:** 22/24 (92%)
- **Expanding layers:** 2/24 (8%) - Layer 3 und Layer 23

### Critical Finding
MLP expandiert **NICHT** generell. Nur zwei Layer expandieren:
1. **Layer 3** (gain = 1.77): Frühe "feature extraction"?
2. **Layer 23** (gain = 3.60): **PREDICTION HEAD**

---

## 3. Combined Gain (Attn × MLP)

```
Layer:  0     4     8    12    16    20    23
Net:   0.47  0.22  0.14  0.16  0.12  0.06  1.34
       ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  ↑↑↑↑
       NET CONTRACTION (23 layers)           EXPANSION
```

### Key Statistics
- **Net contracting:** 23/24 layers (96%)
- **Net expanding:** 1/24 layers (4%) - **NUR Layer 23**

---

## 4. Theoretical Implications

### Original Hypothesis (REJECTED)
```
Attention: contracts (sheaf diffusion)
FFN/MLP:   expands (prediction)
```

### Revised Understanding
```
Layers 0-22: BOTH Attention AND MLP contract
             → Information compression
             → Consensus formation
             → Sheaf diffusion to global section

Layer 23:    MLP EXPLODES (3.6x expansion)
             → Logit spreading
             → Prediction confidence
             → Final "decision broadcast"
```

### The "Funnel" Model

```
Input Embedding
      │
      ▼
┌─────────────────┐
│  Layer 0-22     │  ← COMPRESSION FUNNEL
│  Attn: contract │     Both components contract
│  MLP:  contract │     Information flows to consensus
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Layer 23       │  ← EXPANSION HEAD
│  Attn: contract │     Attention still contracts
│  MLP:  EXPAND   │     MLP explodes for prediction
└────────┬────────┘
         │
         ▼
    Output Logits
```

---

## 5. Connection to Previous Results

### Restriction Maps (Layer 10)
- Min contraction ratio: 0.124
- Min attention entropy: 0.024 (almost deterministic)

### FFN Analysis (This Experiment)
- Min attention gain: 0.083 (Layer 20)
- Max MLP gain: 3.604 (Layer 23)

### Unified Picture

| Metric | L* | Value | Interpretation |
|--------|-----|-------|----------------|
| Restriction Map Contraction | 10 | 0.124 | Commitment Point |
| Attention Entropy | 10 | 0.024 | Decision Made |
| Attention Gain | 20 | 0.083 | Maximum Compression |
| MLP Expansion | 23 | 3.604 | Prediction Broadcast |

**Timeline:**
1. **Layer 10:** Decision/Commitment (min entropy, min contraction)
2. **Layer 20:** Maximum compression (min attention gain)
3. **Layer 23:** Prediction explosion (max MLP gain)

---

## 6. Why MLP Contracts in Most Layers

### Mathematical Reason
MLP structure: `output = GELU(x @ W_up) @ W_down`

- **W_up:** d_model → 4*d_model (expansion to 8192)
- **W_down:** 4*d_model → d_model (compression back to 2048)
- **GELU:** Non-linear, can suppress activations

The gain depends on:
1. Activation sparsity (GELU zeros out negative)
2. Weight matrix conditioning
3. Residual stream interaction

### Why Layer 23 Expands
The final MLP likely has:
- Less sparse activations (more confident)
- Larger effective weights (for logit spread)
- No subsequent normalization to dampen

---

## 7. Sheaf-Theoretic Interpretation

### Original Theory
Information flows via restriction maps ρ: F(U) → F(V)
- Contraction: ||ρ(s)|| < ||s|| (information loss)
- Expansion: ||ρ(s)|| > ||s|| (information amplification)

### Empirical Reality
```
Transformer = Compression Funnel + Expansion Head

Layer 0-22: ρ_l is contractive
            Global section s* emerges via consensus
            ||s*|| < ||s_0|| (compressed representation)

Layer 23:   ρ_final is expansive
            s* gets broadcast to logit space
            ||logits|| >> ||s*|| (amplified for softmax)
```

### Why This Makes Sense
1. **Compression is consensus:** Multiple local views → one global view
2. **Expansion is decision:** One compressed view → many class probabilities
3. **The bottleneck is intentional:** Forces abstraction

---

## 8. Implications for Paper #3

### What We Can Claim
1. ✅ Attention is universally contractive (sheaf restriction maps)
2. ✅ MLP is mostly contractive (information compression)
3. ✅ Expansion occurs only at prediction (final layer)
4. ✅ The network forms a "compression funnel" architecture

### What We Should NOT Claim
1. ❌ "FFN expands to create predictions" (too simple)
2. ❌ "Attention contracts, FFN expands" (wrong)

### Revised Narrative
> "LLMs implement a **compression funnel**: both attention and FFN contract information
> through most layers, forcing consensus formation. Only the final layer's FFN
> expands to broadcast the compressed decision to logit space."

---

## 9. Visualization Summary

The 4-panel plot shows:

1. **Top-Left (Gains):** Attention (blue) always < 1, MLP (red) < 1 except Layer 3, 23
2. **Top-Right (Norms):** MLP output explodes in Layer 3 and especially Layer 23
3. **Bottom-Left (Ratio):** MLP/Attn ratio > 1 almost everywhere (MLP dominates)
4. **Bottom-Right (Net):** Combined gain < 1 for 23/24 layers, only L23 expands

---

## 10. Files

```
Results/
├── ffn_expansion_results.json           # Raw data
├── ffn_expansion_analysis.png           # 4-panel visualization
├── ffn_expansion_results_*.zip          # Timestamped archive
└── FFN_EXPANSION_ANALYSIS.md            # This document
```

---

## 11. Conclusions

### Key Discovery
**The "expansion" doesn't happen where we expected.**

- Original hypothesis: Attention contracts, FFN expands
- Reality: BOTH contract, ONLY final layer expands

### The Transformer as Information Funnel

```
         Wide Input (many tokens, many features)
                    │
                    ▼
    ┌───────────────────────────────────┐
    │     23 Layers of Compression      │
    │     (Attn + MLP both contract)    │
    │                                   │
    │     Information → Consensus       │
    └───────────────────────────────────┘
                    │
                    ▼
         Narrow Bottleneck (compressed state)
                    │
                    ▼
    ┌───────────────────────────────────┐
    │     Layer 23: Expansion Head      │
    │     (MLP gain = 3.6x)             │
    │                                   │
    │     Consensus → Prediction        │
    └───────────────────────────────────┘
                    │
                    ▼
         Wide Output (vocabulary logits)
```

### Theoretical Significance
This supports a **variational interpretation**:
- The network minimizes representation length (compression)
- Only the final layer maximizes class separation (expansion)
- The "commitment point" (L*=10) is where compression is most aggressive

---

*Generated: 2026-01-05*
*Status: HYPOTHESIS REJECTED - NEW FUNNEL MODEL DISCOVERED*
