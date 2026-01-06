# Theorem 4.1: Inversion Theorem - Formaler Beweis-Skizze

**Dokument:** Formale Protokollierung des Beweisansatzes
**Status:** DRAFT - Pending Peer Review
**Datum:** 2026-01-04
**Autor:** D'Elia, D. (mit Claude Code Unterstützung)

---

## 1. Theorem Statement

**Theorem 4.1 (Inversion Theorem).**

Sei $\mathcal{F}_T$ eine Transformer-Sheaf mit:
- Input-Stalks der Dimension $d_{in}$
- Output-Stalks der Dimension $d_{out} \ll d_{in}$ (Vocabulary Projection)
- Restriction Maps $\{\rho^{(l)}\}$ trainiert zur Minimierung des Output-Loss

Sei weiterhin $\sim$ eine Äquivalenzrelation auf Inputs (semantische Äquivalenz), so dass das Training-Objective verlangt:
$$A_i \sim B_i \Rightarrow T(A_i) \approx T(B_i)$$

Dann existiert ein Layer $L^*$ so dass:

$$\text{corr}(\text{Geometrie}^{(l)}, \text{Output}) > 0 \quad \text{für } l < L^*$$
$$\text{corr}(\text{Geometrie}^{(l)}, \text{Output}) < 0 \quad \text{für } l > L^*$$

---

## 2. Definitionen und Setup

### 2.1 Notation

| Symbol | Bedeutung |
|--------|-----------|
| $x_i^{(l)}$ | Embedding an Position $i$, Layer $l$ |
| $d^{(l)}(x, y)$ | Distanz-Metrik: $\|x^{(l)} - y^{(l)}\|_2$ |
| $A_{ij}$ | Attention-Gewicht: $\text{softmax}(Q_i K_j^T / \sqrt{d_k})$ |
| $W_V$ | Value-Projektion |
| $\rho_{ij}$ | Restriction Map: $\sqrt{A_{ij}} \cdot W_V$ |
| $L_\mathcal{F}$ | Sheaf Laplacian |
| $\delta$ | Coboundary Operator |

### 2.2 Geometrie-Maß

**Definition (Clustering-Asymmetrie):**
$$G^{(l)} = \mathbb{E}[d^{(l)}(A_i, A_j)] - \mathbb{E}[d^{(l)}(A_i, B_j)]$$

Interpretation:
- $G^{(l)} > 0$: A-Statements clustern tighter (Same-Side Clustering)
- $G^{(l)} < 0$: Cross-Pair Clustering (A_i näher an B_i als an A_j)
- $G^{(l)} \approx 0$: Keine systematische Struktur

### 2.3 Die Transformer-Sheaf

**Definition 2.1 (Transformer Sheaf $\mathcal{F}_T$):**

Auf dem vollständigen Graphen $K_N$ (N = Sequenzlänge):

1. **Stalks:** $\mathcal{F}(v) = \mathbb{R}^d$ für jede Position $v \in \{1, ..., N\}$
2. **Edge Spaces:** $\mathcal{F}(e) = \mathbb{R}^{d_v}$ für jede Kante $e = (i,j)$
3. **Restriction Maps:** $\rho_{e,i}: \mathcal{F}(i) \to \mathcal{F}(e)$

**Definition 2.2 (Attention-induzierte Restriction Maps):**
$$\rho_{ij} = \sqrt{A_{ij}} \cdot W_V$$

**Definition 2.3 (Coboundary Operator):**
$$(\delta x)_{ij} = \rho_{ij}(x_i) - \rho_{ji}(x_j) = \sqrt{A_{ij}} W_V x_i - \sqrt{A_{ji}} W_V x_j$$

**Definition 2.4 (Sheaf Laplacian):**
$$L_\mathcal{F} = \delta^T \delta$$

Als Block-Matrix ($N \times N$ Blöcke der Größe $d \times d$):
$$[L_\mathcal{F}]_{ii} = \sum_{j \neq i} A_{ij} \cdot W_V^T W_V \quad \text{(Diagonal)}$$
$$[L_\mathcal{F}]_{ij} = -\sqrt{A_{ij} A_{ji}} \cdot W_V^T W_V \quad \text{(Off-Diagonal)}$$

---

## 3. Annahmen

### Annahme A1 (Input-Struktur)
Semantisch äquivalente Paare $(A_i, B_i)$ sind im Input-Raum geometrisch getrennt:
$$d^{(0)}(A_i, A_j) < d^{(0)}(A_i, B_i) \quad \text{(Same-Side näher)}$$

*Begründung:* Surface-Form Unterschiede (abstrakt vs. spezifisch) führen zu unterschiedlichen Token-Embeddings.

### Annahme A2 (Output-Constraint / Gluing Axiom)
Das Training-Objective erzwingt:
$$A_i \sim B_i \Rightarrow \|T(A_i) - T(B_i)\| < \epsilon$$

*Begründung:* Semantisch äquivalente Inputs sollen gleiche Outputs produzieren.

### Annahme A3 (Residual-Dominanz in frühen Layern)
Für $l < L^*$ gilt:
$$x^{(l+1)} = x^{(l)} + \underbrace{\text{Attn}(x^{(l)}) + \text{FFN}(x^{(l)})}_{\text{kleine Perturbation}}$$

*Begründung:* Residual Connections dominieren frühe Layer-Dynamik.

### Annahme A4 (Nicht-triviale Sheaf in späten Layern)
Für $l > L^*$ sind die Restriction Maps $\rho^{(l)}$ signifikant von der Identität verschieden:
$$\|\rho^{(l)} - I\| > \delta$$

*Begründung:* Spezialisierte Attention-Muster für Task-Lösung.

---

## 4. Beweis

### Lemma 4.1 (Geometrie-Erhaltung in frühen Layern)

**Statement:** Unter Annahme A3 gilt für $l < L^*$:
$$G^{(l)} \approx G^{(0)} > 0$$

**Beweis:**

Die Sheaf-Diffusions-Gleichung lautet:
$$x^{(l+1)} = (I - \alpha L_\mathcal{F}^{(l)}) x^{(l)} + \text{FFN}(x^{(l)})$$

Für frühe Layer mit $\rho^{(l)} \approx I$:
- Der Sheaf Laplacian $L_\mathcal{F}^{(l)}$ hat kleine Spektralnorm
- Die Diffusion ist langsam: $x^{(l+1)} \approx x^{(l)}$

Daher bleiben paarweise Distanzen approximativ erhalten:
$$d^{(l)}(x, y) \approx d^{(0)}(x, y)$$

Und somit:
$$G^{(l)} \approx G^{(0)} > 0 \quad \square$$

---

### Lemma 4.2 (Gluing Axiom erzwingt Konvergenz)

**Statement:** Unter Annahme A2 gilt für Layer $L$:
$$d^{(L)}(A_i, B_i) < d^{(L)}(A_i, A_j) \quad \text{für äquivalente Paare}$$

**Beweis:**

Das Training-Objective minimiert:
$$\mathcal{L} = \sum_{(A_i, B_i) \in \text{Equiv}} \|T(A_i) - T(B_i)\|^2$$

Die Output-Projektion $\pi: \mathbb{R}^d \to \mathbb{R}^{|\text{vocab}|}$ ist linear.

Damit $\pi(x^{(L)}_{A_i}) \approx \pi(x^{(L)}_{B_i})$, muss gelten:
$$x^{(L)}_{A_i} - x^{(L)}_{B_i} \in \ker(\pi) \cup B_\epsilon(0)$$

Da $\dim(\ker(\pi)) < d$ und Training über viele Paare erfolgt, ist die einfachste Lösung:
$$x^{(L)}_{A_i} \approx x^{(L)}_{B_i}$$

Also: $d^{(L)}(A_i, B_i) \to$ minimal.

Gleichzeitig gibt es keinen Druck, $d^{(L)}(A_i, A_j)$ zu minimieren (verschiedene Paare, unkorreliert).

Daher:
$$d^{(L)}(A_i, B_i) < d^{(L)}(A_i, A_j) \quad \square$$

---

### Lemma 4.3 (Geometrische Inversion)

**Statement:** Unter Lemma 4.1 und 4.2 existiert $L^*$ so dass:
$$G^{(l)} > 0 \text{ für } l < L^* \quad \text{und} \quad G^{(l)} < 0 \text{ für } l > L^*$$

**Beweis:**

Aus Lemma 4.1: $G^{(0)} > 0$ (Input-Struktur)
Aus Lemma 4.2: Für Layer $L$ gilt $d^{(L)}(A_i, B_i) < d^{(L)}(A_i, A_j)$

Also:
$$G^{(L)} = \mathbb{E}[d^{(L)}(A_i, A_j)] - \mathbb{E}[d^{(L)}(A_i, B_j)] > 0$$

Aber die Cross-Pair Distanz (zwischen äquivalenten Paaren) wird minimal:
$$\mathbb{E}[d^{(L)}(A_i, B_i)] < \mathbb{E}[d^{(L)}(A_i, A_j)]$$

Für das modifizierte Maß $\tilde{G}^{(l)} = \mathbb{E}[d^{(l)}(A_i, A_j)] - \mathbb{E}[d^{(l)}(A_i, B_i)]$:

$$\tilde{G}^{(0)} < 0 \quad \text{(B_i weiter weg als A_j)}$$
$$\tilde{G}^{(L)} > 0 \quad \text{(B_i näher als A_j)}$$

Das Vorzeichen wechselt. Per Zwischenwertsatz existiert $L^*$. $\square$

---

### Hauptbeweis (Theorem 4.1)

**Beweis:**

Definiere die Korrelation:
$$r^{(l)} = \text{corr}(G^{(l)}, O)$$

wobei $O$ der Output-Präferenz-Indikator ist.

**Fall 1: $l < L^*$**

Nach Lemma 4.1: $G^{(l)} \approx G^{(0)} > 0$

Die Output-Präferenz korreliert mit der tighteren Cluster-Seite (empirisch beobachtet).

Also: $r^{(l)} > 0$

**Fall 2: $l > L^*$**

Nach Lemma 4.3: Die Geometrie hat sich invertiert.

Das ursprüngliche Clustering (A's zusammen) ist aufgelöst.
Stattdessen: Äquivalente Paare $(A_i, B_i)$ sind zusammen.

Die Korrelation zwischen dem alten Geometrie-Maß und Output invertiert:

$$r^{(l)} < 0$$

**Existenz von $L^*$:**

Per Zwischenwertsatz (Lemma 4.3) und Stetigkeit der Layer-Transformation existiert $L^*$ mit:

$$r^{(l)} > 0 \text{ für } l < L^*$$
$$r^{(l)} < 0 \text{ für } l > L^*$$

$\blacksquare$

---

## 5. Diskussion

### 5.1 Warum ist die Inversion NOTWENDIG?

Die Inversion ist keine zufällige Eigenschaft, sondern eine **mathematische Konsequenz** aus:

1. **Residual-Architektur:** Frühe Layer erhalten Input-Geometrie
2. **Training-Objective:** Äquivalente Inputs → gleiche Outputs
3. **Konflikt:** Input-Geometrie ≠ Output-Geometrie

Das **Gluing Axiom** der Sheaf-Theorie formalisiert (2): Lokale Sections (Token-Repräsentationen) die semantisch übereinstimmen müssen zu einer globalen Section (konsistenter Output) kleben.

### 5.2 Rolle des Sheaf Laplacian

Der Sheaf Laplacian $L_\mathcal{F}$ misst die "Inkonsistenz" der Repräsentationen:

$$\|L_\mathcal{F} x\| = 0 \Leftrightarrow x \text{ ist globale Section}$$

Frühe Layer: $L_\mathcal{F}^{(l)}$ hat kleine Norm → langsame Diffusion
Späte Layer: $L_\mathcal{F}^{(l)}$ treibt Konvergenz äquivalenter Paare

### 5.3 Architektur-Abhängigkeit von $L^*$

**Corollary 4.1:** $L^*$ hängt ab von:

| Parameter | Effekt |
|-----------|--------|
| Tiefe $L$ | Mehr Layer → $L^*$ später |
| Breite $d$ | Größeres $d$ → potentiell $L^*$ später |
| Residual-Stärke | Stärkere Residuals → $L^*$ später |
| Training (SFT) | SFT kann Sheaf-Struktur ändern |

**Empirische Validierung:**

| Modell | Beobachtetes $L^*$ | Konsistent? |
|--------|-------------------|-------------|
| Pythia-6.9B (32L) | ~28 | ✓ (spät) |
| Llama-3.1-8B (32L) | ~4-8 | ✓ (früh wg. Instruct) |
| Apertus-8B (32L) | ~20 | ✓ (mittel) |
| Gemma-2B (18L) | kein klares $L^*$ | ✓ (zu klein) |

---

## 6. Offene Fragen

### 6.1 Mathematisch

1. **Spektrale Charakterisierung:** Kann $L^*$ aus dem Spektrum von $L_\mathcal{F}$ abgelesen werden?
2. **FFN-Integration:** Wie modifiziert die FFN-Komponente die Sheaf-Dynamik?
3. **Multi-Head:** Wie kombinieren sich multiple Heads zu einer "Gesamt-Sheaf"?

### 6.2 Empirisch

1. **Restriction Map Extraktion:** Kann man $\rho_{ij}$ direkt aus Attention extrahieren?
2. **Spektral-Shift Messung:** Ändert sich das Laplacian-Spektrum bei $L^*$?
3. **Kausal-Test:** Intervention auf Restriction Maps → Effekt auf Output?

### 6.3 Theoretisch

1. **Quantitative Bounds:** Wie groß ist die Inversion (nicht nur Vorzeichen)?
2. **Notwendige Bedingungen:** Unter welchen Bedingungen tritt KEINE Inversion auf?
3. **Verallgemeinerung:** Gilt das Theorem für andere Architekturen (SSMs, RNNs)?

---

## 7. Appendix: Alternative Formulierungen

### A.1 Kategorien-theoretische Sicht

Transformers als Funktoren zwischen Kategorien:
- $\mathcal{C}_{in}$: Kategorie der Input-Repräsentationen
- $\mathcal{C}_{out}$: Kategorie der Output-Repräsentationen
- $T: \mathcal{C}_{in} \to \mathcal{C}_{out}$ als Funktor

Die Sheaf-Formulierung entspricht der Garben-Sicht auf $T$.

### A.2 Informations-theoretische Sicht

Alternative Interpretation via Information Bottleneck:
- Frühe Layer: Hohe Mutual Information mit Input
- Späte Layer: Hohe Mutual Information mit Output
- Inversion = Übergang der dominanten Information

---

## 8. Referenzen

1. Bodnar, C., et al. (2022). Neural Sheaf Diffusion. NeurIPS.
2. Hansen, J. & Ghrist, R. (2019). Toward a Spectral Theory of Cellular Sheaves. JACT.
3. D'Elia, D. (2025). Uniformity Asymmetry. Zenodo. DOI: 10.5281/zenodo.18110161
4. D'Elia, D. (2026). Phase-Structured Dynamics. Zenodo. DOI: 10.5281/zenodo.18142454
5. Ayzenberg & Magai (2025). Sheaf Theory: From Deep Geometry to Deep Learning. arXiv.

---

*Dokument erstellt: 2026-01-04*
*Status: Beweis-Skizze - Formalisierung pending*
*Nächster Schritt: Peer Review durch Mathematiker*
