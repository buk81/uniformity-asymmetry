# Zenodo Upload: Description & Metadata

## Title

Thermodynamic Constraints in Transformer Architectures: A Sheaf-Theoretic Perspective

## Authors

D'Elia, Davide (IU International University of Applied Sciences)*

*Independent research project. Institutional affiliation for identification only.

## Description (for Zenodo "Description" field)

```
We present empirical evidence for thermodynamic-like constraints governing information flow in transformer architectures. Analysis of residual stream dynamics across 23+ models from 7 independent research labs (2022–2024) reveals three quantitative scaling laws:

1. **Kleiber's Law for Transformers** (r = −0.81, p = 0.014; Pythia family): Maximum gain scales as G_max = 10^(1/L), constraining deeper networks toward thermodynamic neutrality.

2. **Training Heritage Dominance** (p < 0.001): Training methodology determines thermodynamic behavior more strongly than architecture—EleutherAI models show 80% dampening while Meta/OpenAI show 100% expansion.

3. **Spectral Signature Correspondence**: The ratio ||W_V||/||W_O|| predicts dampening vs. expansion with 10× magnitude differences between labs.

We validate these laws through direct computation of the Sheaf Laplacian using an efficient O(n² + d²) trace algorithm, demonstrating 26× magnitude differences between model families (GPT-2: 62,696 vs. OPT-125m: 2,368).

**Additional contributions:**
- Dimensional Crowding Theory: Head density ρ = H/d_head mechanistically explains the Pythia anomaly
- Thermodynamic Invariance: RLHF modulates magnitude (up to 50%) but cannot invert sign
- Unified cross-architecture benchmark establishing the hierarchy: Pythia (0.80) < Mistral (1.11) < LLaMA (1.48) < Gemma (2.31)

**Core finding:** Thermodynamic character is determined by pretraining geometry and cannot be overwritten by fine-tuning. The hierarchy is Heritage > Geometry > Scale.

This work provides actionable design principles for practitioners selecting base models and contributes to the theoretical understanding of transformer dynamics.

---
Keywords: transformer architectures, thermodynamic constraints, sheaf theory, residual stream, mechanistic interpretability, scaling laws
```

## Metadata

| Field | Value |
|-------|-------|
| **Resource Type** | Preprint |
| **Publication Date** | 2026-01-06 |
| **Language** | English |
| **License** | CC BY 4.0 |
| **Access** | Open Access |

## Keywords (comma-separated)

```
transformer architectures, thermodynamic constraints, sheaf theory, residual stream dynamics, mechanistic interpretability, scaling laws, language models, Pythia, GPT-2, OPT, training heritage
```

## Related Identifiers

| Relation | Identifier | Type |
|----------|------------|------|
| Is supplement to | 10.5281/zenodo.18110161 | DOI |
| Is supplement to | 10.5281/zenodo.18142454 | DOI |

## Subjects

- Computer Science - Machine Learning
- Computer Science - Computation and Language
- Mathematics - Category Theory

## Files to Upload

1. `thermodynamic_constraints_v3.8.pdf` (compiled LaTeX)
2. `Figures/` (all 8 PNG files, zipped)
3. `code/` (experiment scripts, optional)

## Notes

- This is Paper 3 in a series (Paper 1: Uniformity Asymmetry, Paper 2: Phase-Structured Dynamics)
- All experiments timestamped via OpenTimestamps on Bitcoin blockchain
- Reproducible with PYTHONHASHSEED=42
