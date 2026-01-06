# Figure Implementation Prompt - ECHTE DATEN

**Gib diesen Prompt an Grok, Gemini und ChatGPT. Jede KI soll Python-Code für die 3 Figuren erstellen.**

---

## PROMPT START

Erstelle **produktionsreifen Python-Code** für 3 wissenschaftliche Figuren.

### Anforderungen:
- Nur `matplotlib` (kein seaborn, kein plotly)
- Publication-quality (300 DPI, klare Labels)
- Speichere als PNG
- Konsistente Farbpalette über alle Figuren

### Farbschema (verbindlich):
```python
COLORS = {
    'EleutherAI': '#E74C3C',   # Rot (Dampener)
    'Meta': '#3498DB',          # Blau
    'BigScience': '#27AE60',    # Grün
    'TII': '#F39C12',           # Orange
    'StabilityAI': '#1ABC9C',   # Türkis
}
```

---

## FIGUR 1: Kleiber's Law (Hero-Shot)

**Typ:** Scatter Plot mit Fit-Linie

**Daten (Pythia Familie, 8 Modelle):**
```python
kleiber_data = {
    'model': ['70m', '160m', '410m', '1b', '1.4b', '2.8b', '6.9b', '12b'],
    'layers': [6, 12, 24, 16, 24, 32, 32, 36],
    'gain': [1.201, 1.157, 0.978, 1.216, 1.005, 0.927, 0.994, 0.986],
    'is_dampening': [False, False, True, False, False, True, True, True]
}
# Correlation: r = -0.812, p = 0.014
```

**Design-Vorgaben:**
- X-Achse: Inverse Depth (1/L), Bereich 0 bis 0.20
- Y-Achse: Mean Residual Gain, Bereich 0.88 bis 1.25
- Theoretische Linie: G_max = 10^(1/L) (gestrichelt, grau)
- Horizontale Linie bei G = 1.0 (Neutral)
- Marker: Dreieck runter (▼) für Dampener (rot), Dreieck hoch (△) für Expander (blau)
- Labels mit Pfeilen zu jedem Punkt (Modellname)
- Correlation r und p im Plot (Box unten-links)
- Titel: "Figure 1: Kleiber's Law for Transformers"

---

## FIGUR 2: Training Heritage (Violin/Strip Plot)

**Typ:** Strip Plot mit Mittelwert-Marker

**Daten (16 Modelle, 5 Labs):**
```python
heritage_data = {
    'EleutherAI': {
        'models': ['pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'gpt-neo-1.3b', 'gpt-j-6b'],
        'gains': [1.004, 0.927, 0.994, 0.896, 1.137],
        'mean': 0.992, 'dampen_pct': 80
    },
    'Meta': {
        'models': ['opt-125m', 'opt-350m', 'opt-1.3b', 'opt-2.7b', 'opt-6.7b'],
        'gains': [1.000, 1.078, 1.090, 1.080, 1.263],
        'mean': 1.102, 'dampen_pct': 20
    },
    'BigScience': {
        'models': ['bloom-560m', 'bloom-1b1', 'bloom-1b7', 'bloom-3b'],
        'gains': [0.993, 1.066, 1.097, 1.283],
        'mean': 1.110, 'dampen_pct': 25
    },
    'TII': {
        'models': ['falcon-7b'],
        'gains': [1.027],
        'mean': 1.027, 'dampen_pct': 0
    },
    'StabilityAI': {
        'models': ['stablelm-3b'],
        'gains': [1.084],
        'mean': 1.084, 'dampen_pct': 0
    }
}
```

**Design-Vorgaben:**
- X-Achse: Labs (sortiert nach mean gain: EleutherAI → BigScience)
- Y-Achse: Residual Gain, Bereich 0.85 bis 1.30
- Horizontale Linie bei G = 1.0
- Hintergrund-Shading: Blau unter 1.0 (Dampening Zone), Rot über 1.0 (Expansion Zone)
- Jeder Punkt = ein Modell (mit Jitter für Sichtbarkeit)
- Mittelwert pro Lab als horizontale Linie oder großer Marker
- Unter X-Achse: Lab-Signatur ("DAMPENER" / "EXPANDER")
- Titel: "Figure 2: Training Heritage Determines Thermodynamic Signature"

---

## FIGUR 3: Spectral Signature (Scatter)

**Typ:** Scatter Plot

**Daten (7 Modelle, 4 Labs):**
```python
spectral_data = {
    'model': ['pythia-160m', 'pythia-410m', 'gpt-neo-125M', 'opt-125m', 'opt-350m', 'bloom-560m', 'gpt2'],
    'lab': ['EleutherAI', 'EleutherAI', 'EleutherAI', 'Meta', 'Meta', 'BigScience', 'OpenAI'],
    'W_V_mean': [22.15, 16.93, 25.80, 1.61, 2.75, 3.78, 8.95],
    'W_O_mean': [2.39, 2.46, 70.08, 2.02, 3.43, 2.56, 22.51],
    'behavior': ['DAMPEN', 'DAMPEN', 'DAMPEN', 'EXPAND', 'EXPAND', 'EXPAND', 'EXPAND']
}
# Key finding: 10x difference in ||W_V|| between EleutherAI (16-26) and Meta (1.6-2.8)
```

**Design-Vorgaben:**
- X-Achse: Mean ||W_V|| (Spectral Norm), Log-Skala empfohlen
- Y-Achse: Mean ||W_O|| (Spectral Norm), Log-Skala empfohlen
- Farbe: Nach Lab (COLORS dict)
- Marker: Kreis für Dampener, Quadrat für Expander
- Labels bei jedem Punkt (Modellname)
- Annotation-Box: "10x difference in ||W_V||" mit Pfeil
- Optional: Vertikale gestrichelte Linien bei W_V = 5 und W_V = 15 (Regime-Grenzen)
- Titel: "Figure 3: Spectral Signatures Predict Thermodynamic Behavior"

---

## Output-Format

Liefere **einen vollständigen Python-Code** der:
1. Alle 3 Figuren in einem Script erstellt
2. Als `fig1_kleiber.png`, `fig2_heritage.png`, `fig3_spectral.png` speichert
3. Ohne externe Dependencies außer matplotlib und numpy läuft

**Bonus:** Erstelle auch eine kombinierte 1x3 Figur (`fig_combined.png`) für Präsentationen.

---

## PROMPT END
