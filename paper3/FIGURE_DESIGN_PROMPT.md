# Figure Design Competition Prompt

**Kopiere diesen Prompt an Grok, Gemini und ChatGPT. Vergleiche ihre Vorschläge.**

---

## PROMPT START

Ich schreibe ein wissenschaftliches Paper für arXiv/NeurIPS-Level über eine Entdeckung in Transformer-Architekturen. Ich brauche deine Hilfe beim Design der optimalen Figuren für maximalen wissenschaftlichen Impact.

### Paper Summary

**Titel:** "Thermodynamic Constraints in Transformer Architectures: A Sheaf-Theoretic Perspective"

**Core Discovery:** Wir haben entdeckt, dass Transformer "thermodynamische Signaturen" haben - manche Modelle dämpfen Information (Gain < 1), andere expandieren sie (Gain > 1). Diese Signatur wird primär durch das Training Lab bestimmt, nicht durch die Architektur.

### Die 3 Hauptergebnisse (alle statistisch signifikant)

**Finding 1: Kleiber's Law für Transformer (H25)**
- Korrelation: r = -0.878, p = 0.004
- Tiefere Netzwerke (mehr Layer) haben niedrigeren Gain
- Formel: G_max ≈ 10^(1/L)
- Analogie zu biologischem Kleiber's Law (Metabolismus ~ Masse^0.75)

**Daten (Pythia Familie, 8 Modelle):**
```
Model        Layers   Gain    Type
pythia-70m      6     1.201   Expander
pythia-160m    12     1.157   Expander
pythia-410m    24     0.978   Dampener
pythia-1b      16     1.216   Expander
pythia-1.4b    24     1.005   Expander (borderline)
pythia-2.8b    32     0.927   Dampener
pythia-6.9b    32     0.994   Dampener (borderline)
pythia-12b     36     0.986   Dampener
```

**Finding 2: Training Heritage Dominance (H26)**
- Das Training Lab bestimmt die thermodynamische Signatur stärker als Architektur
- Fisher's exact test: p < 0.001

**Daten (16 Modelle, 5 Labs):**
```
Lab           Models  Mean Gain  Dampen%  Signature
EleutherAI       5      0.945     80%     DAMPENER
TII              1      1.027      0%     NEUTRAL
StabilityAI      1      1.084      0%     EXPANDER
Meta             5      1.102     20%     EXPANDER
BigScience       4      1.110     25%     EXPANDER
```

**Finding 3: Spectral Signature Correspondence (H27)**
- Die Spektralnormen der Weight-Matrizen W_V und W_O predicten das Verhalten
- 10x Unterschied in ||W_V|| zwischen Labs!

**Daten (7 Modelle):**
```
Model         Lab          ||W_V||   ||W_O||   Ratio   Behavior
pythia-160m   EleutherAI    22.15     2.39     9.27    DAMPEN
pythia-410m   EleutherAI    16.93     2.46     6.88    DAMPEN
gpt-neo-125M  EleutherAI    25.80    70.08     0.37    DAMPEN
opt-125m      Meta           1.61     2.02     0.80    EXPAND
opt-350m      Meta           2.75     3.43     0.80    EXPAND
bloom-560m    BigScience     3.78     2.56     1.48    EXPAND
gpt2          OpenAI         8.95    22.51     0.40    EXPAND
```

### Methodologische Besonderheit

Wir haben einen kritischen Measurement-Artifact entdeckt: Die finale LayerNorm verfälscht Gain-Messungen. Nach Korrektur verbesserte sich unsere Validation Accuracy von 43.75% auf 100% innerhalb der Pythia-Familie.

### Deine Aufgabe

Designe **3-4 Figuren** für maximalen wissenschaftlichen Impact. Beachte:

1. **Klarheit über Komplexität** - Jede Figur sollte EINE klare Botschaft haben
2. **Reviewer-Perspektive** - Was würde einen skeptischen Reviewer überzeugen?
3. **Visual Hierarchy** - Figure 1 ist die wichtigste ("Hero Figure")
4. **Kein Overload** - Lieber weniger Figuren, dafür perfekt

### Bitte liefere für jede Figur:

1. **Titel** der Figur
2. **Typ** (Scatter, Bar, Heatmap, Panel, etc.)
3. **X-Achse / Y-Achse / Farb-Encoding**
4. **Key Message** - Was soll der Leser sofort verstehen?
5. **Warum diese Figur?** - Strategische Begründung

### Bonus-Fragen:

- Sollte Figure 1 ein Combo-Panel sein (2x2) oder ein einzelnes starkes Bild?
- Brauchen wir eine "Methods"-Figur (Layer-wise dynamics)?
- Wie visualisiert man den 10x Unterschied in ||W_V|| am besten?

---

## PROMPT END

---

## Nach dem Wettbewerb

Wenn alle 3 KIs geantwortet haben:

1. Vergleiche die Vorschläge
2. Wähle die beste Strategie (oder kombiniere)
3. Gib der Winner-KI die echten Daten zum Erstellen

**Bewertungskriterien:**
- Klarheit der Visualisierungsstrategie
- Wissenschaftliche Überzeugungskraft
- Kreativität/Originalität
- Praktische Umsetzbarkeit
