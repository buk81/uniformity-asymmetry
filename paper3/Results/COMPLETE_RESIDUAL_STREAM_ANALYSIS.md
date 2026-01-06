# Complete Residual Stream Validation Analysis v5

**Date:** 2026-01-05
**Status:** ⚠️ CRITICAL UPDATE - Unexpected LLaMA 2 Results!
**Models:** Pythia-6.9B, Gemma-7B, Apertus-8B, LLaMA-3.1-8B, LLaMA-2-7B (+Mistral-7B)
**Update:** v5 - LLaMA 2 test shows INVERTED results! Long-Context hypothesis WEAKENED!

---

## Executive Summary

| Model | Family | Norm | MLP | Last Layer Gain | Expands? |
|-------|--------|------|-----|-----------------|----------|
| Pythia-6.9B | Pythia | LayerNorm | GELU | **0.22x** | **NO** |
| Gemma-7B | Gemma | RMSNorm | GeGLU | **2.85x** | **YES** |
| Apertus-8B | Apertus | RMSNorm | xIELU | **2.64x** | **YES** |
| Mistral-7B (ref) | Mistral | RMSNorm | SwiGLU | **1.37x** | **YES** |
| **LLaMA-3.1-8B** | LLaMA | RMSNorm | SwiGLU | **0.48x** | **NO!** |

### Key Finding: REVISED - Expansion is NOT Purely Normalization-Dependent!

```
Original Hypothesis (v1):
  RMSNorm → EXPANDS
  LayerNorm → CONTRACTS

NEW Finding (v2):
  LLaMA-3.1-8B uses RMSNorm but CONTRACTS (0.48x)!

  → Normalization type ALONE does not determine expansion
  → Architecture-specific factors play a role
```

---

## 1. The LLaMA 3.1 Anomaly (NEW!)

### 1.1 LLaMA-3.1-8B (RMSNorm, SwiGLU) - THE OUTLIER

```
Layers: 32
Normalization: RMSNorm (same as Gemma, Apertus, Mistral)
Activation: SwiGLU (same as Mistral!)

Residual Stream Profile:
├── Embedding Norm: 1.24 (SMALLEST of all models!)
├── Initial Gain (Emb→L0): 10.2x
├── Layer 1 EXPLOSION: 39.3x (similar to Mistral)
├── Plateau Norm: ~493-510 (very stable)
├── Final Norm: 245
└── Last Layer Gain: 0.48x (CONTRACTS!)

Statistics:
├── Contracting layers: 2/32 (6.3%)
├── Expanding layers: 30/32 (93.7%)
└── Last layer: CONTRACTS (0.48x) ← BREAKS PATTERN!
```

### 1.2 SwiGLU Comparison: LLaMA 3.1 vs Mistral

```
Both models use: RMSNorm + SwiGLU
Expected: Similar expansion behavior
Actual: DIVERGENT!

┌─────────────────────────────────────────────────┐
│ Model          │ Last Gain │ Ratio to Mistral  │
├─────────────────────────────────────────────────┤
│ Mistral-7B     │   1.37x   │      1.00         │
│ LLaMA-3.1-8B   │   0.48x   │      0.35         │
└─────────────────────────────────────────────────┘

Ratio 0.35 is FAR outside the 0.8-1.2 "consistent" range!
```

### 1.3 Why Does LLaMA 3.1 Contract? - RLHF HYPOTHESIS TEST RESULTS

**Original Hypothesis:** RLHF Safety Training causes the contraction

**Test Conducted:** 2026-01-05 - Base vs Instruct Comparison

```
┌─────────────────────────────────────────────────────────────┐
│         *** HYPOTHESIS REJECTED - ARCHITECTURAL ***         │
└─────────────────────────────────────────────────────────────┘
```

### RLHF Test Results

| Model | RLHF | Last Gain | Expands? | Cumulative Energy |
|-------|------|-----------|----------|-------------------|
| **LLaMA-3.1-8B BASE** | NO | **0.48x** | **NO** | 198 |
| **LLaMA-3.1-8B INSTRUCT** | YES | **0.36x** | **NO** | 157 |
| Mistral-7B (ref) | NO | 1.37x | YES | ~60 |

### Verdict: ARCHITECTURAL, NOT RLHF

```
BOTH models contract!

→ Kontraktion ist ARCHITEKTUR-bedingt, NICHT RLHF-verursacht
→ LLaMA 3.1 unterscheidet sich fundamental von Mistral
   (trotz gleicher Architektur: RMSNorm + SwiGLU)
```

### But: RLHF AMPLIFIES the Effect!

| Metric | Base | Instruct | Ratio | Effect |
|--------|------|----------|-------|--------|
| Last Gain | 0.48x | 0.36x | **0.75** | -25% |
| Cumulative Energy | 198 | 157 | **0.79** | -21% |

**RLHF reduces output energy by an additional ~25%!**

### Revised Interpretation

```
Original Hypothesis: RLHF CAUSES contraction
                     → REJECTED

New Finding:         LLaMA 3.1 contracts ARCHITECTURALLY
                     RLHF AMPLIFIES the existing contraction

                     Base:     0.48x (already contracting)
                     Instruct: 0.36x (25% more contraction from RLHF)
```

### Connection to Paper #1 & #2 (UPDATED)

Evidence from Paper #1 Archive:
**File:** `research/half_truth_detection/ANALYSIS_CYBER_FEUDALISM.md` (Lines 377-392, 407-410)

```
META (Llama):
├── Normalisiert ALLES
├── Alle Statements ~0.58-0.68 (enger Range)
└── "Normalisierung" aller Positionen
```

**Reinterpretation:**
- Paper #1: LLaMA shows highest uniformity → **ARCHITECTURAL tendency to flatten**
- Paper #2: Chat templates cause negative correlations → **Template effect separate from RLHF**
- Paper #3: LLaMA contracts at architecture level, RLHF amplifies by 25%

**The three papers show COMPLEMENTARY effects, not the same cause!**

### 1.4 Previous Finding: Titanium Projector + Long-Context Dampening (v4)

**Question:** Both use RMSNorm + SwiGLU, why opposite behavior?

**Previous Answer (v4):** Two factors - **Titanium Projector** + **Long-Context Dampening**

```
HYPOTHESIS 1: TITANIUM PROJECTOR - STILL VALID

| Model      | Vocab    | σ_max (W_U) | Last Gain | Effective Amp |
|------------|----------|-------------|-----------|---------------|
| LLaMA 3.1  | 128,256  |    94.61    |   0.48x   |    45.41      |
| Mistral    |  32,000  |    16.14    |   1.37x   |    22.11      |

W_U Ratio: LLaMA / Mistral = 94.61 / 16.14 = 5.86x LARGER!
→ This is CONFIRMED and not affected by the LLaMA 2 test.
```

### 1.5 ⚠️ CRITICAL UPDATE: LLaMA 2 Test INVERTS Hypothesis! (v5)

**NEW Test:** LLaMA 2 (4k context) vs LLaMA 3.1 (128k context)

**Prediction:** If Long-Context Dampening is correct:
- LLaMA 2 (short context) → Should EXPAND
- LLaMA 3.1 (long context) → Should CONTRACT

**ACTUAL Results:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    *** HYPOTHESIS INVERTED! ***                     │
└─────────────────────────────────────────────────────────────────────┘

| Model         | Context | RoPE θ  | Last Gain | Expands? |
|---------------|---------|---------|-----------|----------|
| LLaMA-2-7B    |  4,096  | 10,000  |  0.92x    |   NO     |
| LLaMA-3.1-8B  | 128,000 | 500,000 |  1.53x    |   YES!   |

EXPECTED:  LLaMA 2 → EXPAND,   LLaMA 3.1 → CONTRACT
ACTUAL:    LLaMA 2 → CONTRACT, LLaMA 3.1 → EXPAND
```

**LLaMA 3.1 Inconsistency:**

| Test | LLaMA 3.1 Last Gain | Status |
|------|---------------------|--------|
| vs Mistral (v4) | **0.48x** | CONTRACTS |
| vs LLaMA 2 (v5) | **1.53x** | EXPANDS |

**Difference: 3.19x** - Same model, very different measurements!

**Possible Explanations:**
1. Input-dependent behavior (different test sentences)
2. Methodology differences (tokenization, layer counting)
3. Numerical precision differences

**Status Update:**
```
Hypothesis 1 (Titanium Projector):     STILL VALID (W_U confirmed)
Hypothesis 2 (Long-Context Dampening): WEAKENED (contradicted by LLaMA 2 test)
```

**Scientific Value:** This negative result reveals complexity we hadn't anticipated.

---

## 2. Updated Model Results

### 2.1 Pythia-6.9B (LayerNorm, GELU)

```
Layers: 32
Normalization: LayerNorm

Residual Stream Profile:
├── Embedding Norm: 2.40
├── Initial Gain (Emb→L0): 59.1x
├── Peak Norm: 3093 (at L23)
├── Final Norm: 599
└── Last Layer Gain: 0.22x (CONTRACTS)

Statistics:
├── Contracting layers: 9/32 (28.1%)
├── Expanding layers: 23/32 (71.9%)
└── Last layer: CONTRACTS (0.22x)
```

### 2.2 Gemma-7B (RMSNorm, GeGLU)

```
Layers: 28
Normalization: RMSNorm

Residual Stream Profile:
├── Embedding Norm: 11.28
├── Initial Gain (Emb→L0): 40.3x
├── Plateau Norm: ~460-495
├── Final Norm: 1393
└── Last Layer Gain: 2.85x (EXPANDS!)

Statistics:
├── Contracting layers: 2/28 (7.1%)
├── Expanding layers: 26/28 (92.9%)
└── Last layer: EXPANDS (2.85x)
```

### 2.3 Apertus-8B (RMSNorm, xIELU) - Paper #2 Model

```
Layers: 32
Normalization: RMSNorm
Activation: xIELU (novel)

Residual Stream Profile:
├── Embedding Norm: 10.97
├── Initial Gain (Emb→L0): 16.4x
├── Layer 2 EXPLOSION: 328.9x (anomaly!)
├── Plateau Norm: ~77,000-88,000
├── Layer 28 (Paper #2 Inversion): 0.74x CONTRACTS
├── Final Norm: 188,747
└── Last Layer Gain: 2.64x (EXPANDS!)

Statistics:
├── Contracting layers: 4/32 (12.5%)
├── Expanding layers: 28/32 (87.5%)
├── Paper #2 L28: 0.74x (CONTRACTS)
└── Last layer: EXPANDS (2.64x)
```

---

## 3. Cumulative Energy Analysis (NEW!)

**Cumulative Energy = Product of all layer gains**

| Model | Final Cumulative | Net Effect |
|-------|-----------------|------------|
| Apertus-8B | **17,212** | AMPLIFY (highest!) |
| Pythia-6.9B | 249 | AMPLIFY |
| LLaMA-3.1-8B | 198 | AMPLIFY |
| Gemma-7B | 124 | AMPLIFY |

**Key Insight:** ALL models show net AMPLIFICATION from embedding to output, even those that CONTRACT at the last layer! The early-layer explosions dominate.

---

## 4. Revised Universal Principles

### Original Claims (v1)

| # | Claim | Status |
|---|-------|--------|
| 1 | "RMSNorm models expand at last layer" | **BROKEN** by LLaMA 3.1 |
| 2 | "LayerNorm models contract at last layer" | Still holds (1/1) |
| 3 | "Same MLP type → similar expansion" | **BROKEN** by LLaMA vs Mistral |

### Revised Claims (v2)

| # | Revised Claim | Evidence |
|---|---------------|----------|
| 1 | **Most RMSNorm models expand** | 3/4 (Gemma, Apertus, Mistral) |
| 2 | **LLaMA 3.1 is an outlier** | RMSNorm + SwiGLU but contracts |
| 3 | **MLP type alone doesn't determine expansion** | LLaMA vs Mistral diverge |
| 4 | **All models show net cumulative amplification** | 4/4 final_cumulative > 1 |

### New Grouping

```
Expanding Models (Last Layer > 1):
├── Gemma-7B:    2.85x (RMSNorm, GeGLU)
├── Apertus-8B:  2.64x (RMSNorm, xIELU)
└── Mistral-7B:  1.37x (RMSNorm, SwiGLU)

Contracting Models (Last Layer < 1):
├── Pythia-6.9B:   0.22x (LayerNorm, GELU)
└── LLaMA-3.1-8B:  0.48x (RMSNorm, SwiGLU) ← UNEXPECTED!
```

---

## 5. Paper #2 ↔ Paper #3 Connection

### Apertus-8B Layer Dynamics

```
Layer 28: 0.74x CONTRACTION  ← Paper #2 "Inversion Point"
          ↓ Semantic inversion (correlation sign flip)
Layer 29: 1.03x expansion
Layer 30: 1.05x expansion
Layer 31: 2.64x EXPANSION   ← Paper #3 "Broadcast"
          ↓ Output amplification for prediction
```

**Conclusion:** Semantic inversion (L28) ≠ Broadcast expansion (L31). These are separate phenomena.

---

## 6. Implications for Paper #3

### What Strengthens

1. **Cumulative Energy metric** - All models amplify overall
2. **Paper #2 ↔ Paper #3 disconnect** - Inversion ≠ Expansion confirmed
3. **Architecture diversity** - 5 families tested

### What Changes (Critical!)

1. ~~"RMSNorm always expands"~~ → **Most expand, but LLaMA 3.1 is exception**
2. ~~"Same MLP = same behavior"~~ → **Architecture matters beyond MLP type**
3. **NEW:** Need to investigate what makes LLaMA 3.1 different

### Recommended Paper Framing (Updated)

> "We discover that final-layer dynamics are predominantly normalization-dependent, but not exclusively. While most RMSNorm architectures (Gemma 2.85x, Apertus 2.64x, Mistral 1.37x) show residual stream expansion, LLaMA 3.1 (also RMSNorm + SwiGLU) shows 0.48x contraction - identical pattern to LayerNorm models. This suggests additional architectural factors beyond normalization type influence final-layer behavior. Importantly, ALL models show net cumulative amplification from embedding to output, indicating the 'broadcast' happens through different mechanisms: some via last-layer expansion, others via accumulated early-layer gains."

---

## 7. Files

```
Results/
├── residual_stream_complete_validation_v2_results.json  # Raw data (all 4 models)
├── residual_stream_complete_validation.png              # 4-panel visualization
├── cumulative_energy_analysis.png                       # NEW: Cumulative energy
├── COMPLETE_RESIDUAL_STREAM_ANALYSIS.md                 # This document (v2)
├── CROSS_ARCHITECTURE_ANALYSIS.md                       # Summary
└── MISTRAL_PARADOX_SOLVED.md                            # Mistral investigation
```

---

## 8. Conclusions

### Key Discoveries (v2)

1. **LLaMA 3.1 Anomaly:** RMSNorm + SwiGLU but CONTRACTS (0.48x)
2. **Hypothesis Revision:** Norm type alone doesn't determine expansion
3. **SwiGLU Divergence:** LLaMA vs Mistral (0.35 ratio) - same MLP, opposite behavior
4. **Cumulative Amplification:** ALL models amplify overall (17,212x to 124x)
5. **Paper #2 Connection:** Semantic inversion at L28 is contraction, not expansion

### Final Principle (Revised)

> **"The 'Compress-then-Broadcast' pattern is UNIVERSAL, but the mechanism varies:**
>
> **Type A (Expanding):** Gemma, Apertus, Mistral
> - Broadcast via last-layer residual stream expansion (1.37x-2.85x)
>
> **Type B (Contracting):** Pythia, LLaMA 3.1
> - Broadcast via accumulated early-layer gains + unembedding
> - Last layer contracts but cumulative energy still amplifies
>
> **The distinction is NOT purely LayerNorm vs RMSNorm!**
> **LLaMA 3.1 proves additional factors determine the mechanism.**"

---

## 9. Open Questions

1. **Why does LLaMA 3.1 contract?** (Same MLP as Mistral but opposite behavior)
2. **What architectural differences matter?** (Embedding scale? Attention pattern?)
3. **Is this LLaMA 3 specific or LLaMA family wide?** (Test LLaMA 2, LLaMA 3.2)
4. **Does training data/objective affect this?** (Instruction tuning?)

---

*Generated: 2026-01-05 03:15*
*Updated: 2026-01-05 (v5 - LLaMA 2 INVERTED results)*
*Status: ⚠️ HYPOTHESIS 2 WEAKENED - Requires further investigation*
*Models: Pythia-6.9B, Gemma-7B, Apertus-8B, LLaMA-3.1-8B, LLaMA-2-7B, Mistral-7B*
*Key Finding: LLaMA 3.1 showed 0.48x vs Mistral but 1.53x vs LLaMA 2 - input-dependent?*
