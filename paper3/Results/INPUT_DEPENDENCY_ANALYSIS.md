# Input-Dependency Thermodynamics Analysis

**Date:** 2026-01-05
**Status:** EXPERIMENTELL VALIDIERT - Mit ÜBERRASCHENDEN Befunden!
**Model:** LLaMA-3.1-8B

---

## Executive Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│   H18 (Input-Dependency): CONFIRMED but REFINED with SURPRISING NUANCES │
└─────────────────────────────────────────────────────────────────────────┘

Original Hypothesis (v6):
├── Gain ∝ Output Entropy
├── Factual (low entropy) → CONTRACTS
├── Ambiguous (high entropy) → EXPANDS
└── Schachtelsatz (constrained) → CONTRACTS

EXPERIMENTAL RESULTS:
├── ✅ Input-Dependency confirmed: Gain varies with prompt type
├── ❌ Gain ∝ Entropy: WIDERLEGT (Pearson r = 0.21, p = 0.74)
├── 🔧 Schachtelsatz → HIGHEST gain (0.80)! Grammar parsing is EXPENSIVE!
└── 📚 Ambiguous → LOWEST entropy (3.00)! Strong priors for philosophical!
```

---

## Experimental Design

### 5 Prompt Types Tested

| # | Type | Prompt | Expected |
|---|------|--------|----------|
| 1 | **Factual** | "The capital of France is" | Low Gain, Low Entropy |
| 2 | **Syntactic** | "The agreement, which, notwithstanding the fact that it was signed only yesterday, effectively binds all parties immediately, stipulates that" | Low Gain? (grammar constrains) |
| 3 | **Ambiguous** | "The true meaning of happiness is often found in" | High Gain, High Entropy |
| 4 | **Nonsense** | "Table sky run blue jump quickly under over" | High Gain, High Entropy |
| 5 | **Original** | "The quick brown fox jumps over the lazy dog." | Medium (reference) |

---

## Results

### Raw Data

| Prompt Type | Last-Layer Gain | Output Entropy | Top Token | Top Prob |
|-------------|-----------------|----------------|-----------|----------|
| **Factual** | **0.478** | 4.03 | "a" | 17.5% |
| **Syntactic** | **0.802** ⬆️ | 4.85 | "the" | 34.7% |
| **Ambiguous** | 0.563 | **3.00** ⬇️ | "the" | 53.7% |
| **Nonsense** | 0.596 | **7.34** ⬆️ | "the" | 5.9% |
| **Original** | 0.609 | 4.32 | "The" | 41.3% |

### Correlation Analysis

```
Entropy ↔ Gain Correlation:
────────────────────────────
Pearson r  = 0.21  (p = 0.74) → NOT SIGNIFICANT
Spearman ρ = 0.60  (p = 0.28) → Moderate trend, but n=5 too small

Verdict: Simple linear relationship REJECTED
```

---

## Key Findings

### 1. Schachtelsatz Paradox: INVERTED!

**Original Hypothesis:**
> Nested grammar CONSTRAINS the search space → LLMs find it EASY → LOW gain

**Actual Result:**
> Schachtelsatz has **HIGHEST gain (0.802)** of all prompt types!

**Interpretation:**
Grammar PARSING is computationally EXPENSIVE, not "easy" for LLMs. The model must:
- Track nested dependencies
- Maintain agreement across clauses
- Parse complex syntactic structure

→ **Grammar processing requires MORE energy, not less!**

### 2. Ambiguous: The Anti-Intuitive Result!

**Original Hypothesis:**
> "Meaning of happiness" is open-ended → HIGH entropy → model "explores"

**Actual Result:**
> **LOWEST entropy (3.00)** of all prompt types! Top token "the" at **53.7%** confidence!

**Interpretation:**
The model has **strong learned priors** for philosophical completions:
- "The true meaning of happiness is often found in **the**..."
- This is a common pattern in training data
- Model is CONFIDENT, not uncertain!

→ **"Open-ended" to humans ≠ "Open-ended" to LLMs**

### 3. Nonsense: As Expected

- Entropy: **7.34** (Maximum) ✓
- Top token probability: 5.9% (lowest) ✓
- Gain: 0.596 (moderate)

→ **Confusion/fallback mode confirmed**

---

## Revised Model: Task-Type Dependent Computation

The simple "Gain ∝ Entropy" hypothesis is **TOO SIMPLISTIC**!

### Two Orthogonal Axes

```
           HIGH ENTROPY (uncertain)
                ↑
    Nonsense ●  │  (7.34, 0.60)
                │
    Syntactic   │● (4.85, 0.80) ← GRAMMAR PARSING!
    Original    │● (4.32, 0.61)
    Factual  ●  │  (4.03, 0.48)
                │
    Ambiguous ● │  (3.00, 0.56) ← STRONG PRIORS!
                │
           LOW ENTROPY (certain)
    ←────────────────────────→
    LOW GAIN         HIGH GAIN
```

### Task-Type → Computation Mode Mapping

| Task Type | Gain | Entropy | Model Mode | Description |
|-----------|------|---------|------------|-------------|
| **Factual** | Low (0.48) | Medium (4.03) | 🔍 **Retrieval** | Fast pattern match, "I know this" |
| **Syntactic** | **HIGH (0.80)** | Medium (4.85) | 🔧 **Parsing** | Grammar processing, expensive! |
| **Ambiguous** | Medium (0.56) | **LOW (3.00)** | 📚 **Prior-Based** | Strong learned pattern, confident |
| **Nonsense** | Medium (0.60) | **HIGH (7.34)** | ❓ **Fallback** | Confusion, no good options |
| **Original** | Medium (0.61) | Medium (4.32) | 🔄 **Standard** | Balanced processing |

---

## Physical Interpretation

### Updated Energy Formula

**NOT:** Energy ∝ Entropy (too simple)

**INSTEAD:** Energy = f(Task_Type, Parsing_Complexity, Prior_Strength)

```
FACTUAL:   Known fact → Retrieval mode → LOW energy (cruise control)
SYNTACTIC: Complex grammar → Parsing mode → HIGH energy (gear shift!)
AMBIGUOUS: Philosophical → Prior-based mode → MEDIUM energy (autopilot)
NONSENSE:  Chaos → Fallback mode → MEDIUM energy (confusion)
```

### LLM Difficulty ≠ Human Intuition

But DIFFERENTLY than originally hypothesized:

| Task Type | Human Difficulty | LLM Difficulty (Gain) |
|-----------|------------------|----------------------|
| Factual | Easy | **Easy** (0.48) ✓ |
| Schachtelsatz | **HARD** | **HARD** (0.80) ✓ |
| Ambiguous | Variable | **Easy** (confident!) |
| Nonsense | Confusing | Confusing |

**Key Insight:** Grammar is hard for BOTH humans AND LLMs!
The "Schachtelsatz Paradox" (grammar constrains search) was WRONG.

---

## 🚗 Das "Bremspedal-Gesetz" (Gemini's Reframing)

**Die entscheidende Einsicht:** Wir messen nicht Expansion vs. Kontraktion, sondern **MODULATION DER DÄMPFUNG**!

### LLaMA 3.1 steht IMMER auf der Bremse

Egal was reinkommt, der Gain ist < 1.0. Aber der **Grad der Dämpfung** variiert:

```
┌─────────────────────────────────────────────────────────────────┐
│                   LLaMA 3.1: IMMER AUF DER BREMSE               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Factual (0.48x)  ████████████████████████░░░░░░  VOLLBREMSUNG  │
│  Ambiguous (0.56x) ██████████████████████████░░░░  STARKE BREMSE │
│  Nonsense (0.60x)  ███████████████████████████░░░  BREMSE        │
│  Original (0.61x)  ███████████████████████████░░░  BREMSE        │
│  Syntactic (0.80x) █████████████████████████████░  BREMSE GELÖST!│
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│  0.0          0.5          1.0 (Neutral)              1.5       │
│               ▲                    ▲                   ▲         │
│          LLaMA 3.1            Mistral              Gemma        │
│          (Dämpfung)           (Inertia)          (Instabil)     │
└─────────────────────────────────────────────────────────────────┘
```

### Das Architektur-Bias + Input-Modulation Modell

**Architektur setzt den BASE LEVEL:**

| Architektur | Base Level | Physik | Analogie |
|-------------|------------|--------|----------|
| **LLaMA 3.1** | < 1.0 | Aktive Dämpfung | Bremspedal immer gedrückt |
| **Mistral** | ≈ 1.0 | Inertia | Nutzt Schwung |
| **Gemma** | > 1.0 | Instabil | Neigt zur Explosion |

**Input MODULIERT um den Base Level:**

```
Komplexität ↑  →  Gain ↑  (weniger bremsen / mehr Gas)

LLaMA:   0.48 ──────► 0.80  ("Bremse lockern")
Mistral: 1.0  ──────► 1.5?  ("Gas geben")
Gemma:   2.8  ──────► ???   ("Explosion kontrollieren")
```

### Der "Plattitüden-Tunnel"

Warum "Meaning of Happiness" (Ambiguous) nicht explodiert:

> "Weil das für ein LLM, das auf 15 Billionen Token trainiert wurde, **keine hohe Entropie** ist. Es hat Millionen solcher Sätze gesehen. Es ist ein Klischee."

**Physik:** Das Modell rutscht sofort in einen tiefen, bekannten "Plattitüden-Tunnel" - nur für MENSCHEN ambiguos, nicht für das Modell!

### Finales Physikalisches Modell

```
System = Architektur_Bias × Input_Modulation

Energie = Base_Level(Architektur) + Δ(Input_Komplexität)

Wobei:
- Base_Level ∈ {<1 (Dämpfung), ≈1 (Inertia), >1 (Instabil)}
- Δ(Komplexität) = Modulation basierend auf Parsing-Anforderung
```

**Das ist die "Thermodynamik der Entscheidung":**
1. **Architektur** bestimmt den Arbeitspunkt (Bremser vs. Raser)
2. **Input** moduliert um diesen Arbeitspunkt
3. **Complexity** (nicht Entropy!) treibt die Modulation

---

## Implications for Paper #3

### 1. Residual Stream Dynamics are State-Dependent ✅

Confirmed: The same architecture produces different gain profiles for different inputs.

### 2. Simple Thermodynamics Model is Incomplete

The "Energy ∝ Uncertainty" model needs refinement:
- Need to account for **computation type** (retrieval vs parsing)
- Need to account for **prior strength** (trained patterns)

### 3. New Research Direction: Parsing Circuits

The high gain for Schachtelsatz suggests dedicated grammar parsing circuits that:
- Activate for complex syntactic structures
- Require significant computational resources
- Operate independently of output entropy

### 4. Prior Confidence vs Output Entropy

"Ambiguous" prompts reveal a disconnect:
- High **semantic** openness (many valid completions)
- Low **learned** entropy (strong training patterns)

The model's training distribution dominates over "logical" uncertainty.

---

## Limitations

1. **n=5 is too small** for statistical significance
2. **Single model** (LLaMA-3.1-8B) - need cross-model validation
3. **Single token** - entropy measured only at next token position
4. **No layer-wise analysis** - only final layer gain measured

---

## Future Work

1. **Expand prompt set** (n > 30 for statistical power)
2. **Cross-model validation** (Pythia, Gemma, Mistral)
3. **Layer-wise gain analysis** for each prompt type
4. **Attention pattern analysis** for Schachtelsatz parsing
5. **Training data correlation** for prior strength

---

## Files

```
Results/
├── input_dependency_thermodynamics.json  # Raw experiment data
├── INPUT_DEPENDENCY_ANALYSIS.md          # This document

notebooks/
└── Input_Dependency_Thermodynamics.ipynb # Colab notebook
```

---

*Generated: 2026-01-05*
*Status: H18 CONFIRMED with SURPRISING NUANCES!*
*Key Finding: Schachtelsatz → HIGHEST gain! Ambiguous → LOWEST entropy!*
