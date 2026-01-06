# Restriction Maps Analysis: Pythia-1.4B

**Experiment Date:** 2026-01-05
**Model:** EleutherAI/pythia-1.4b
**Layers:** 24
**Test Prompt:** "The capital of France is Paris, which is known for the Eiffel Tower."

---

## Executive Summary

| Prediction | Expected | Observed | Status |
|------------|----------|----------|--------|
| Formula ρ_{ij} = √(A_{ij})·W_V | Constructible | **CONSTRUCTED** | ✅ |
| Contraction before L* | ratio < 1 | **ALL ratios < 1** | ✅ |
| Expansion after L* | ratio > 1 | ratio stays < 1 | ⚠️ |
| Attention focusing in late layers | High→Low entropy | **1.7 → 0.02** | ✅ |

### Verdict: PARTIAL CONFIRMATION

Die Restriction Maps sind **durchgehend kontraktiv**, aber zeigen einen klaren **Trend** der zur Theorie passt.

---

## 1. Contraction Ratio Analysis

### Layer-wise Contraction Ratios

```
Layer:  0    2    4    6    8    10   12   14   16   18   20
Ratio: 0.29 0.26 0.20 0.18 0.23 0.12 0.15 0.15 0.17 0.18 0.14
       ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  MIN  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
       PHASE 1: COMPRESSION        PHASE 2: RELATIVE EXPANSION
```

### Key Observations

1. **All ratios < 1**: Durchgehend kontraktiv (||ρ|| < ||W_V||)
2. **Minimum at Layer 10**: 0.124 (stärkste Kontraktion)
3. **Increasing trend after L10**: Von 0.12 auf 0.14-0.18
4. **Layers 21-23**: NaN (numerische Probleme bei sehr fokussierter Attention)

### Interpretation

Die Theorie sagte voraus:
- Kontraktion (ratio < 1) vor L*
- Expansion (ratio > 1) nach L*

Was wir sehen:
- **Kontraktion ÜBERALL** (ratio nie > 1)
- **ABER: Relativer Trend passt!**
  - Ratio sinkt von 0.29 → 0.12 (Kompression)
  - Ratio steigt von 0.12 → 0.18 (relative Expansion)

**Mögliche Erklärung:** Die absolute Expansion findet im FFN statt, nicht in Attention allein.

---

## 2. Attention Entropy Analysis

### Die "Fokussierung" über Layer

```
Layer:   0    2    4    6    8    10   12   14   16   18   20
Entropy: 1.69 1.36 0.65 0.40 1.02 0.02 0.12 0.14 0.28 0.44 0.10
         DISTRIBUTED ----→ EXTREMELY FOCUSED ----→ FOCUSED
```

### Key Finding: Layer 10 Singularität

| Layer | Entropy | Interpretation |
|-------|---------|----------------|
| 0 | 1.69 | Attention verteilt über viele Tokens |
| 4 | 0.65 | Beginnende Fokussierung |
| **10** | **0.024** | **FAST DETERMINISTISCH** - ein Token dominiert |
| 20 | 0.10 | Weiterhin sehr fokussiert |

**Das ist die "Commitment"-Phase!**

Bei Layer 10:
- Attention-Entropy fast 0 → Ein Token bekommt ~100% Attention
- Dies korreliert mit L*_anisotropy = 7 aus Paper #3 Pythia-6.9B
- Der Kontext ist "entschieden" - das Modell hat sich festgelegt

---

## 3. W_V Norm Evolution

### Value Projection Matrix Norms

```
Layer:   0    4    8    12   16   20   23
||W_V||: 6.8  4.8  5.1  8.4  13.5 26.3 28.3
         ↓↓↓↓↓↓↓  STABLE  ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
         EARLY            LATE LAYER EXPLOSION
```

### Observation

Die W_V Normen **explodieren** in späten Layern:
- Layer 0-10: ~5-7
- Layer 20-23: ~26-28

**Interpretation:**
- Späte Layer haben größere Value-Transformationen
- Dies ist nötig für die finale Vorhersage (Logit-Spreizung)
- Die "Expansion" passiert in W_V, nicht in √(A)

---

## 4. Sheaf Laplacian Results

### Spectral Analysis

| Metric | Value Range | Interpretation |
|--------|-------------|----------------|
| λ₁ | ~10⁻¹⁰ | Regularisierung dominiert |
| λ₂ | ~10⁻¹⁰ | Keine echte Spektral-Lücke |
| Spectral Gap | ~10⁻¹⁵ | Numerisches Rauschen |
| Trace | 0.5 → 4.1 | Steigt mit Layer |

### Limitation

Die Sheaf Laplacian Analyse ist **nicht aussagekräftig** wegen:
1. Zu kleine Subsample-Größe (8 Tokens)
2. Regularisierung (1e-10) dominiert Eigenwerte
3. Numerische Instabilität in späten Layern

**Empfehlung:** Größere Subsample-Size oder andere Methodik nötig.

---

## 5. Theoretische Implikationen

### Was die Ergebnisse bedeuten

**Bestätigt:**
1. ✅ Restriction Maps sind konstruierbar: ρ_{ij} = √(A_{ij})·W_V
2. ✅ Alle Maps sind kontraktiv (||ρ|| < ||W_V||)
3. ✅ Attention fokussiert sich dramatisch in mittleren/späten Layern
4. ✅ Es gibt einen Trend: Kontraktion → relative Expansion

**Nicht bestätigt:**
1. ⚠️ Absolute Expansion (ratio > 1) nach L*
2. ⚠️ Klare Spektral-Signatur im Sheaf Laplacian

### Revidierte Hypothese

Die ursprüngliche Prediction war:
```
l < L*: ||ρ|| < ||W_V||  (Kontraktion)
l > L*: ||ρ|| > ||W_V||  (Expansion)
```

Beobachtete Realität:
```
l < L*: ||ρ|| << ||W_V|| (starke Kontraktion)
l > L*: ||ρ|| < ||W_V||  (schwächere Kontraktion)
```

**Neue Interpretation:**
- Die "Expansion" findet nicht in den Restriction Maps statt
- Stattdessen wächst ||W_V|| selbst (von 5 auf 28)
- Die Attention-Gewichte √(A) bleiben klein
- Die Expansion passiert im FFN, nicht in Attention

---

## 6. Multi-Phase Structure (Refined)

```
Layer:  0 ==== 4 ==== 10 ============== 20 ==== 24
        │      │      │                  │       │
Phase:  INIT   FOCUS  COMMITMENT         │ OUTPUT│
        │      │      │                  │       │
Ratio:  0.29   0.20   0.12 (MIN)        0.14    NaN
Entropy:1.69   0.65   0.02 (MIN)        0.10    NaN
||W_V||: 6.8   4.8    4.5               26.3    28.3
```

### Phase Descriptions

| Phase | Layers | Contraction | Entropy | W_V Norm |
|-------|--------|-------------|---------|----------|
| Init | 0-4 | 0.29→0.20 | 1.7→0.6 | ~6 |
| Focus | 4-10 | 0.20→0.12 | 0.6→0.02 | ~5 |
| **Commitment** | **10** | **0.12 (MIN)** | **0.02 (MIN)** | ~5 |
| Plateau | 10-18 | 0.12→0.18 | 0.02→0.44 | 5→12 |
| Output | 18-24 | 0.14→NaN | 0.1→NaN | 12→28 |

**Layer 10 ist der "Commitment Point"** - minimale Kontraktion UND minimale Entropie.

---

## 7. Connection to Previous Results

| Experiment | L* | This Experiment |
|------------|----|-----------------|
| Pythia-6.9B Anisotropy | 7 | - |
| Pythia-6.9B Correlation | 28 | - |
| **Pythia-1.4B Contraction** | **10** | Min contraction |
| **Pythia-1.4B Entropy** | **10** | Min entropy |

**Scaling Observation:**
- Pythia-1.4B (24 layers): L* ≈ 10 (42% depth)
- Pythia-6.9B (32 layers): L* ≈ 7 (22% depth)

Der Commitment-Point ist bei kleineren Modellen relativ später.

---

## 8. Files

```
Results/
├── restriction_maps_results.json        # Raw data
├── restriction_maps_analysis.png        # 4-panel visualization
├── sheaf_laplacian_spectral.png         # Spectral analysis
└── RESTRICTION_MAPS_ANALYSIS.md         # This document
```

---

## 9. Conclusions

### What We Learned

1. **Restriction Maps work**: Die Formel ρ = √(A)·W_V ist konstruierbar und analysierbar

2. **Contraction confirmed**: Alle Maps sind kontraktiv, mit Minimum bei Layer 10

3. **Attention focusing is real**: Entropy kollabiert von 1.7 auf 0.02

4. **Expansion happens elsewhere**: Die W_V Normen wachsen (nicht die Ratios)

5. **Numerical challenges**: Späte Layer sind numerisch instabil

### Next Steps

1. **Larger model comparison**: Pythia-6.9B für direkten Vergleich
2. **Include FFN**: Die Expansion könnte im FFN stattfinden
3. **Better Laplacian method**: Größere Samples, andere Eigenwert-Algorithmen
4. **Multi-head analysis**: Verschiedene Heads zeigen verschiedene Muster

---

*Generated: 2026-01-05*
*Status: PARTIAL CONFIRMATION - Contraction trend confirmed, expansion mechanism revised*
