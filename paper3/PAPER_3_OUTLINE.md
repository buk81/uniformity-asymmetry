# Paper #3 Outline: Sheaf-Theoretic Explanation of Phase-Structured Dynamics

**Working Title:**
*"Transformers as Implicit Sheaf Networks: A Topological Explanation for Phase-Structured Embedding-Output Dynamics"*

**Target Venue:** NeurIPS 2026 / ICLR 2027 (Theoretical Track)

**Status:** DRAFT OUTLINE

---

## Abstract (Draft)

Recent empirical work has demonstrated that embedding-output relationships in large language models exhibit *phase-structured dynamics*: early layers show positive correlation between embedding geometry and output preference, while late layers show systematic inversion. We propose a theoretical explanation grounded in **cellular sheaf theory**.

We formalize transformer architectures as implicit sheaf neural networks, where attention mechanisms define restriction maps between token stalks, and the layer-wise computation corresponds to sheaf diffusion. Under this framework, we prove that:
1. Early layers with near-identity restriction maps preserve input geometry (positive correlation)
2. Late layers with learned non-trivial maps must enforce global consistency via the **gluing axiom**, necessarily inverting local geometric relationships
3. The depth of inversion depends on the sheaf structure induced by the architecture

Our theoretical predictions align with empirical observations across four model families (Pythia, Llama, Apertus, Gemma), providing the first unified mathematical explanation for phase-structured dynamics in LLMs.

---

## 1. Introduction

### 1.1 Motivation

- **Empirical puzzle** (Papers #1, #2): Why does embedding-output correlation invert in late layers?
- **Gap in literature**: Mechanistic interpretability describes *what* happens, not *why*
- **Our contribution**: Sheaf theory provides mathematical *necessity* for inversion

### 1.2 Key Claims

| Claim | Type | Status |
|-------|------|--------|
| Transformers are implicit sheaf networks | Theoretical | To prove |
| Attention defines restriction maps | Definitional | To formalize |
| Gluing axiom → late-layer inversion | Theorem | To prove |
| Inversion depth = f(architecture) | Corollary | To derive |

### 1.3 Paper Structure

1. Background: Sheaf Neural Networks
2. Main Result: Transformers as Sheaf Networks
3. Theorem: Gluing Axiom Implies Inversion
4. Empirical Validation: Phase-Structured Dynamics
5. Predictions & Future Work

---

## 2. Background

### 2.1 Cellular Sheaves (Definitions)

**Definition 2.1 (Cellular Sheaf).**
A cellular sheaf F on a graph G = (V, E) assigns:
- To each node v ∈ V: a vector space F(v) called the *stalk*
- To each edge e = (u,v): linear maps (restriction maps)
  - ρ_{e,u}: F(u) → F(e)
  - ρ_{e,v}: F(v) → F(e)

**Definition 2.2 (Sheaf Laplacian).**
```
L_F = δ^T δ
```
where δ is the coboundary operator:
```
(δx)_e = ρ_{e,u}(x_u) - ρ_{e,v}(x_v)
```

**Definition 2.3 (Global Section).**
A global section is x ∈ ⊕_v F(v) such that L_F x = 0, i.e., locally consistent data that "glues" globally.

### 2.2 Sheaf Neural Networks (Bodnar et al., 2022)

- Standard GNNs assume *trivial sheaf* (all ρ = Identity)
- Neural Sheaf Diffusion learns ρ from data
- Non-trivial sheaves enable better heterophily handling

**Key Result (Bodnar):**
> "Discretised parametric diffusion processes have greater control over asymptotic behaviour when the sheaf is non-trivial."

### 2.3 Categorical View of Attention (Recent Work)

- Self-attention as parametric endofunctor (arXiv:2501.02931)
- Multi-layer = free monad construction
- Query/Key/Value = 1-morphisms in Para(Vect)

---

## 3. Main Result: Transformers as Implicit Sheaf Networks

### 3.1 Setup

**Definition 3.1 (Transformer Sheaf).**
For a transformer with sequence length N and hidden dimension d, define sheaf F_T:

- **Stalks:** F_T(i) = ℝ^d for each position i ∈ {1,...,N}
- **Edges:** Fully connected (attention allows all-to-all communication)
- **Restriction maps:** Defined by attention + projection

### 3.2 Attention as Dynamic Restriction Maps (Gemini/D'Elia Formalization)

**Definition 3.3 (Underlying Topology).**
The underlying space is the complete graph K_N where V = {1,...,N} represents token positions and E represents all-to-all connectivity allowed by attention.

**Definition 3.4 (The Transformer Sheaf).**
A cellular sheaf F_T on K_N consists of:

1. **Stalks:** For every node v ∈ V, the stalk F(v) = ℝ^d is the token embedding space
2. **Interaction Spaces:** For every edge e = (i,j), we have F(e) = ℝ^{d_v} where d_v is the value/head dimension
3. **Restriction Maps:** For each edge e = (i,j):
   - ρ_{e,i}: F(i) → F(e) (source → interaction)
   - ρ_{e,j}: F(j) → F(e) (target → interaction)

**Proposition 3.1 (Attention as Dynamic Restriction Maps).**
A single-head attention mechanism defines data-dependent restriction maps:

```
ρ_{e,ij} = √(A_{ij}) · W_V
```

where A_{ij} = softmax(Q_i K_j^T / √d_k) is the attention weight.

**Proof.**
The attention output for position i is:
```
x'_i = Σ_j A_{ij} · W_V · x_j
```

Rewriting in sheaf notation with ρ_{ij} = √(A_{ij}) · W_V:
```
x'_i = Σ_j ρ_{ij}^T · ρ_{ij} · x_j
```

This matches the discrete sheaf diffusion equation:
```
x' = (I - L_F) x
```

**Key Distinction (Transport vs. Mixing):**
- Standard GNNs: scalar mixing with A_{ij} ∈ ℝ (trivial sheaf)
- Transformers: **vector-valued transport** with ρ_{ij} ∈ ℝ^{d×d_v}

The restriction map W_V **transforms** data as it moves along edges, not just mixes it. This explains why transformers are more expressive than standard message-passing GNNs.

### 3.3 The Transformer Sheaf Laplacian

**Definition 3.5 (Coboundary Operator).**
The coboundary operator δ: ⊕_v F(v) → ⊕_e F(e) measures consistency error:

```
(δx)_e = ρ_{e,i}(x_i) - ρ_{e,j}(x_j)
```

For attention with maps ρ_{ij} = √(A_{ij}) · W_V:
```
(δx)_{ij} = √(A_{ij}) · W_V · x_i - √(A_{ji}) · W_V · x_j
```

**Definition 3.6 (Transformer Sheaf Laplacian).**
The Sheaf Laplacian is:
```
L_F = δ^T · δ
```

Expanded as block matrix (N×N blocks of size d×d):
```
[L_F]_{ii} = Σ_{j≠i} A_{ij} · W_V^T W_V           (diagonal blocks)
[L_F]_{ij} = -√(A_{ij} A_{ji}) · W_V^T W_V        (off-diagonal blocks)
```

**Proposition 3.2 (Diffusion Equation).**
One transformer attention layer computes:
```
x^{(l+1)} = x^{(l)} - α · L_F^{(l)} · x^{(l)} + FFN(x^{(l)})
         = (I - α·L_F^{(l)}) · x^{(l)} + FFN(x^{(l)})
```

where α is an implicit step size and L_F^{(l)} is the layer-l Sheaf Laplacian.

**Remark (Dynamic vs. Static).**
Unlike standard Sheaf Neural Networks where L_F is fixed or slowly learned, the Transformer Sheaf Laplacian is **recomputed at every layer** based on the current representations x^{(l)}. This is "dynamic sheaf diffusion."

### 3.3 Layer-wise Diffusion

**Proposition 3.2.**
One transformer layer corresponds to one step of sheaf diffusion:
```
x^(l+1) = x^(l) + Attn(x^(l)) + FFN(x^(l))
```

In sheaf notation:
```
x^(l+1) = (I + P^(l)_sheaf + N^(l)) x^(l)
```

where P^(l)_sheaf is the sheaf diffusion operator and N^(l) is the non-linear (FFN) component.

### 3.4 The Full Transformer Sheaf

**Definition 3.2 (Layered Transformer Sheaf).**
The full transformer defines a *sequence* of sheaves {F^(0), F^(1), ..., F^(L)} with:
- Stalks shared across layers (same dimension d)
- Restriction maps layer-dependent: ρ^(l)_{ij}
- Functorial composition: F^(L) ∘ F^(L-1) ∘ ... ∘ F^(1)

---

## 4. Theorem: Gluing Axiom Implies Inversion

### 4.1 The Core Insight

**Intuition:**
- Input space (embeddings): High-dimensional, rich local structure
- Output space (logits): Low-dimensional, globally constrained
- Gluing axiom: Local sections must be globally consistent
- **Projection from rich → constrained necessarily inverts geometry**

### 4.2 Formal Statement

**Theorem 4.1 (Inversion Theorem).**
Let F_T be a transformer sheaf with:
- Input stalks of dimension d_in
- Output stalks of dimension d_out << d_in (vocabulary projection)
- Restriction maps {ρ^(l)} learned to minimize output loss

Then there exists a layer L* such that:
```
corr(geometry^(l), output) > 0  for l < L*
corr(geometry^(l), output) < 0  for l > L*
```

**Proof sketch:**

1. **Early layers (l < L*):**
   - Restriction maps ρ^(l) ≈ I (near-identity, preserving input structure)
   - Sheaf is approximately trivial
   - Diffusion preserves local geometry
   - → Positive correlation with eventual output

2. **Transition (l ≈ L*):**
   - Maps become non-trivial to solve the task
   - Sheaf Laplacian eigenspectrum shifts

3. **Late layers (l > L*):**
   - Gluing axiom requires: δx = 0 (consistency)
   - Output space d_out << d_in
   - Projection π: ℝ^d → ℝ^{vocab} compresses
   - **Key:** Compression that preserves output consistency must *invert* input geometry

   Formally: If local sections cluster by input similarity, but output requires different clustering, the gluing process inverts the relationship.

### 4.3 Corollaries

**Corollary 4.1 (Architecture-Dependent Depth).**
The inversion layer L* depends on:
- Depth (more layers → later L*)
- Width (larger d → potentially later L*)
- Training objective (different tasks → different L*)

**Corollary 4.2 (Boundary Conditions).**
Phase structure may not emerge when:
- Model too shallow (insufficient layers for transition)
- Model too small (cannot learn non-trivial sheaf)
- Training method (SFT may alter sheaf structure)

*This explains Gemma-2B as boundary case.*

---

## 5. Empirical Validation

### 5.1 Predictions from Theory

| Prediction | From Theorem | Testable? |
|------------|--------------|-----------|
| Inversion exists in large models | 4.1 | ✓ (Paper #2) |
| Depth of L* is architecture-specific | Cor 4.1 | ✓ (Paper #2) |
| Small models may lack inversion | Cor 4.2 | ✓ (Gemma) |
| Templates change sheaf structure | 3.2 | ✓ (Llama-Instruct) |

### 5.2 Alignment with Paper #2 Results

| Model | Theory Predicts | Observed | Match? |
|-------|-----------------|----------|--------|
| Pythia-6.9B | Late inversion | L28-32 inversion | ✓ |
| Llama-3.1-8B | Earlier transition | L4-8 transition | ✓ |
| Apertus-8B | Inversion before final | Max at L28, not L32 | ✓ |
| Gemma-2B | Boundary (no inversion) | No significant inversion | ✓ |

### 5.3 New Predictions (Future Work)

1. **Restriction map analysis:**
   Extract A_{ij} W_V from attention, compute actual sheaf Laplacian, verify spectrum shift at L*

2. **Causal test:**
   Intervene on restriction maps at different layers, measure effect on output

3. **Scale curve:**
   Plot L* vs model size, expect monotonic relationship

---

## 6. Discussion

### 6.1 Why This Explanation is Compelling

1. **Mathematical necessity:** Inversion isn't arbitrary—it's required by the gluing axiom
2. **Unifies observations:** Explains all findings from Papers #1 and #2
3. **Makes predictions:** Testable claims about scale, architecture, training
4. **Connects to broader theory:** Links transformers to topological deep learning

### 6.2 Relation to Prior Work

| Work | Contribution | Our Extension |
|------|--------------|---------------|
| Bodnar et al. (2022) | Sheaf GNNs | Sheaf Transformers |
| Self-Attention as Functor (2025) | Categorical view | Sheaf-specific formalization |
| D'Elia (2025, 2026) | Empirical phase structure | Theoretical explanation |

### 6.3 Limitations

1. **Idealized model:** We ignore softmax non-linearity in formal proofs
2. **FFN component:** Not fully captured in sheaf framework
3. **Empirical validation:** Need to extract actual restriction maps

### 6.4 Broader Implications

**For Mechanistic Interpretability:**
- Phase-structured dynamics is *mathematically necessary*, not a bug
- Interventions must account for sheaf structure

**For AI Safety:**
- Early-layer "honesty" may be inverted by gluing process
- Claims about model beliefs must specify layer

**For Architecture Design:**
- Sheaf-aware transformers could control inversion depth
- Potential for more interpretable models

---

## 7. Conclusion

We have shown that transformer architectures can be formalized as implicit sheaf neural networks, where attention mechanisms define restriction maps and layer-wise computation corresponds to sheaf diffusion. Under this framework, the empirically observed phase-structured dynamics—early positive correlation inverting to late negative correlation—emerges as a mathematical consequence of the **gluing axiom**: the requirement that local representations must be globally consistent.

Our theory explains:
- Why inversion occurs (gluing axiom)
- Why depth varies by architecture (different sheaf structures)
- Why small models may not show inversion (insufficient complexity for non-trivial sheaf)
- Why templates change the pattern (alter restriction maps)

**Central Claim:**
> *Late-layer inversion in transformers is not an empirical curiosity but a topological necessity arising from the sheaf structure implicit in attention-based architectures.*

---

## References (Key)

1. Bodnar, C., et al. (2022). Neural Sheaf Diffusion. NeurIPS.
2. Hansen, J. & Ghrist, R. (2019). Toward a Spectral Theory of Cellular Sheaves. JACT.
3. D'Elia, D. (2025). Uniformity Asymmetry. Zenodo.
4. D'Elia, D. (2026). Phase-Structured Dynamics. Zenodo.
5. Ayzenberg & Magai (2025). Sheaf Theory: From Deep Geometry to Deep Learning. arXiv.
6. Self-Attention as Parametric Endofunctor (2025). arXiv.

---

## Appendix Ideas

A. **Full Proof of Theorem 4.1**
B. **Spectral Analysis of Transformer Sheaf Laplacian**
C. **Algorithm for Extracting Restriction Maps from Attention**
D. **Extended Empirical Results**

---

## TODO for Paper #3

- [ ] Formalize Definition 3.1 rigorously
- [ ] Prove Proposition 3.1 (attention = restriction maps)
- [ ] Prove Theorem 4.1 (main inversion theorem)
- [ ] Implement restriction map extraction
- [ ] Compute sheaf Laplacian for Pythia layers
- [ ] Verify spectral shift at inversion layer
- [ ] Write introduction with proper framing
- [ ] Consult with category theory expert?

---

*Draft created: 2026-01-04*
*Based on: Papers #1, #2 + Sheaf Neural Network literature review*
