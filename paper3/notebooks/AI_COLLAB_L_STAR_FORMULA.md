# L* Formula Improvement Challenge - AI Collaboration

**Goal:** Reduce prediction error from 10% to <5%

---

## 1. THE PROBLEM

In transformer language models, there is a characteristic **phase transition point L*** where:
- Before L*: Information gathering phase (correlation to output INCREASES)
- After L*: Information synthesis phase (correlation to output DECREASES)

This is the **"Bentov Point"** - named after the inversion in embedding-output correlation.

**Current Challenge:** Predict L* from observable model properties.

---

## 2. COMPLETE DATASET (8 Models)

| Model | L (Layers) | G (Gain) | L* (Empirical) | Behavior | L*/L Ratio |
|-------|-----------|----------|----------------|----------|------------|
| pythia-160m | 12 | 1.157 | 7 | DAMPEN | 0.583 |
| pythia-410m | 24 | 0.978 | 16 | DAMPEN | 0.667 |
| pythia-1b | 16 | 1.216 | 15 | DAMPEN | 0.938 |
| pythia-2.8b | 32 | 0.927 | 26 | DAMPEN | 0.813 |
| pythia-6.9b | 32 | 0.994 | 21 | DAMPEN | 0.656 |
| opt-125m | 12 | 1.263 | 8 | EXPAND | 0.667 |
| gpt2 | 12 | 1.05 | 9 | EXPAND | 0.750 |
| gemma-2b | 18 | 1.0 | 17 | UNKNOWN | 0.944 |

### Additional Metrics (3 Models with W_V data)

| Model | ||W_V||_F | ||W_V||_2 | Entropy | Behavior |
|-------|---------|---------|---------|----------|
| opt-125m | 15.95 | 1.61 | 0.66 | EXPAND |
| gpt2 | 88.15 | 8.95 | 0.79 | EXPAND |
| pythia-160m | 44.44 | 22.15 | 0.87 | DAMPEN |

### Extended Model Metadata

| Model | n_heads | Lab | Trace (MH) |
|-------|---------|-----|------------|
| pythia-160m | 12 | EleutherAI | 18,887 |
| pythia-410m | 16 | EleutherAI | 11,326 |
| pythia-1b | 8 | EleutherAI | 278M |
| pythia-2.8b | 32 | EleutherAI | NaN |
| pythia-6.9b | 32 | EleutherAI | NaN |
| opt-125m | 12 | Meta | 2,368 |
| gpt2 | 12 | OpenAI | 62,696 |
| gemma-2b | 8 | Google | 286M |

---

## 3. DEFINITIONS

**Gain (G):** Ratio of residual stream norms
```
G = ||x^(L-1)||_2 / ||x^(0)||_2
```
- G < 1: DAMPEN (norms decrease through network)
- G > 1: EXPAND (norms increase through network)

**L* (Transition Point):** Layer where trace derivative changes sign
```
L* = argmax_l [d/dl Tr(L_F)]  (empirical)
```

**Behavior Classes:**
- **DAMPEN:** EleutherAI models (Pythia family) - 80% show dampening
- **EXPAND:** OpenAI/Meta models - 100% show expansion
- **UNKNOWN:** Cannot classify from gain alone

---

## 4. CURRENT FORMULAS (Ranked by Error)

### Formula v1 (Original, G-based)
```
L* = (L/2) × (1 + tanh(5×(G-1)))
```
**Error: 25.0%** - Fails because G doesn't fully determine L*

### Formula v2 (Simple Baseline)
```
L* = L/2
```
**Error: 25.2%** - Too simplistic

### Formula v3 (Behavior + Size)
```
L* = L × (α + β×L)

DAMPEN: α = 0.55, β = 0.008
EXPAND: α = 0.60, β = 0.010
```
**Error: 10.0%** - Current best, but requires behavior classification

### Formula v4 (Optimized Linear, behavior-agnostic)
```
L* = L × (0.544 + 0.0087×L)
```
**Error: 12.3%** - Worse than behavior-aware

### Formula v5 (Log-depth)
```
L* = L × (0.31 + 0.17×ln(L))
```
**Error: 13.8%** - Logarithmic scaling

---

## 5. KEY OBSERVATIONS

### Observation 1: L*/L Ratio Increases with Depth
```
Correlation(L, L*/L) = 0.47

Shallow (L≈12): L* ≈ 0.6-0.7L
Deep (L≈32):    L* ≈ 0.7-0.8L
```

### Observation 2: Behavior Matters More Than Gain
```
DAMPEN models: L* tends to be earlier (0.65-0.85L)
EXPAND models: L* tends to be later (0.67-0.75L)
```

### Observation 3: Outliers
- **pythia-1b:** L*/L = 0.938 (very late transition)
- **gemma-2b:** L*/L = 0.944 (very late transition)
- **pythia-6.9b:** L*/L = 0.656 (early for 32-layer model)

### Observation 4: W_V Norm Correlation (3 models)
```
Higher ||W_V||_F → Higher trace → Different L* dynamics?
OPT: 15.95 (low)
GPT-2: 88.15 (high)
Pythia: 44.44 (medium)
```

---

## 6. THEORETICAL CONSTRAINTS

Any improved formula should ideally:

1. **Be interpretable** - Coefficients should have physical meaning
2. **Respect bounds** - 0 < L* < L always
3. **Scale correctly** - Work for L ∈ [12, 32] and extrapolate to L ∈ [48, 80]
4. **Use available inputs** - L, G, behavior (ideally without behavior if possible)
5. **Have theoretical motivation** - Connect to information theory, thermodynamics, or sheaf theory

### Physical Intuition

The Bentov Point represents where the network transitions from:
- **Early layers:** Building local features, increasing complexity
- **Late layers:** Compressing to output, reducing redundancy

This is analogous to:
- **Thermodynamics:** Maximum entropy point
- **Information theory:** Rate-distortion tradeoff
- **Sheaf theory:** Gluing condition satisfaction

---

## 7. CHALLENGE QUESTIONS

1. **Can you find a formula with <5% error using only L and G?**

2. **Is there a theoretical derivation from information theory?**
   - E.g., L* = L × f(H) where H is some entropy measure

3. **What functional form best captures the depth scaling?**
   - Linear: L* = aL + bL²
   - Log: L* = L(a + b·ln(L))
   - Power: L* = aL^c
   - Sigmoid: L* = L/(1 + exp(-a(L-b)))

4. **Can the outliers (pythia-1b, gemma-2b) be explained?**
   - Both have n_heads = 8 (fewer than others)
   - Both have late L* (>90% of depth)

5. **Is there a connection to attention head count?**
   ```
   n_heads=8:  Late L* (0.94)
   n_heads=12: Medium L* (0.67)
   n_heads=16+: Varies (0.65-0.81)
   ```

---

## 8. EVALUATION CRITERIA

**Primary metric:** Mean absolute percentage error (MAPE)
```
MAPE = (1/n) × Σ |L*_pred - L*_emp| / L × 100%
```

**Target:** MAPE < 5%

**Current leaderboard:**
| Formula | MAPE | Inputs |
|---------|------|--------|
| Behavior+Size | 10.0% | L, behavior |
| Linear (Opt) | 12.3% | L |
| Log (Opt) | 13.8% | L |
| v1 (G-based) | 25.0% | L, G |
| L/2 Baseline | 25.2% | L |

---

## 9. RAW DATA FOR ANALYSIS

```python
data = [
    {"model": "pythia-160m", "L": 12, "G": 1.157, "L_star": 7, "behavior": "DAMPEN", "n_heads": 12},
    {"model": "pythia-410m", "L": 24, "G": 0.978, "L_star": 16, "behavior": "DAMPEN", "n_heads": 16},
    {"model": "pythia-1b", "L": 16, "G": 1.216, "L_star": 15, "behavior": "DAMPEN", "n_heads": 8},
    {"model": "pythia-2.8b", "L": 32, "G": 0.927, "L_star": 26, "behavior": "DAMPEN", "n_heads": 32},
    {"model": "pythia-6.9b", "L": 32, "G": 0.994, "L_star": 21, "behavior": "DAMPEN", "n_heads": 32},
    {"model": "opt-125m", "L": 12, "G": 1.263, "L_star": 8, "behavior": "EXPAND", "n_heads": 12},
    {"model": "gpt2", "L": 12, "G": 1.05, "L_star": 9, "behavior": "EXPAND", "n_heads": 12},
    {"model": "gemma-2b", "L": 18, "G": 1.0, "L_star": 17, "behavior": "UNKNOWN", "n_heads": 8},
]

# Additional W_V data (3 models)
wv_data = {
    "opt-125m": {"W_V_frob": 15.95, "entropy": 0.66},
    "gpt2": {"W_V_frob": 88.15, "entropy": 0.79},
    "pythia-160m": {"W_V_frob": 44.44, "entropy": 0.87},
}
```

---

## 10. SUBMIT YOUR FORMULA

Format your response as:

```
## Proposed Formula

L* = [your formula here]

## Parameters
- param1 = value
- param2 = value

## Predictions
| Model | Predicted | Empirical | Error |
...

## MAPE: X.X%

## Theoretical Motivation
[Why this formula makes sense]
```

---

*Created: 2026-01-06*
*Paper: "Thermodynamic Constraints in Transformer Architectures: A Sheaf-Theoretic Perspective"*
*Author: Davide D'Elia*
