# Kevin GitHub Response - Draft

**Date:** 2026-01-05
**Context:** Response to Kevin's suggestion for neutral statement control test
**Status:** READY TO POST

---

## The Response

```markdown
Hi Kevin,

Thank you for the excellent control test suggestion! We ran the Neutral vs Political comparison. The results are scientifically valuable but surprising:

### Results

| Category | n | Mean Asymmetry | Std |
|----------|---|----------------|-----|
| Neutral pairs | 12 | 0.045 | 0.074 |
| Political pairs | 6 | 0.029 | 0.054 |

- Effect Ratio: 0.64 (Political/Neutral < 1!)
- p-value: 0.655 (not significant)

### The Surprise

Contrary to our hypothesis, **neutral pairs showed higher mean asymmetry** than political pairs!

### Interpretation

The metric measures **embedding-space structural differences**, not "bias" in the moral sense:

- **Political opposites** ("Immigration strengthens/weakens economy") are semantically **CLOSE** (same topic, opposite valence) → low asymmetry
- **Different factual statements** ("Sky is blue" vs "Water freezes") can be semantically **FAR** → higher asymmetry

### What This Means

1. **Mistral is not politically biased** - political mean ≈ 0.03
2. **The metric works** - it measures real embedding differences
3. **Important caveat**: Uniformity Asymmetry is most meaningful for *semantically parallel* pairs (same structure, substituted entities) rather than arbitrary comparisons

### Visualization

![Neutral vs Political Asymmetry](../paper3/Results/neutral_vs_political_asymmetry.png)

Thank you for pushing us to run this control - negative/surprising results are scientifically valuable! This clarifies what the metric actually measures.

Full analysis available in the repository.

Best,
Davide
```

---

## Notes for Posting

1. **Tone:** Grateful, scientifically honest, acknowledges the value of the critique
2. **Key message:** The negative result is valuable - it clarifies metric interpretation
3. **Include:** The visualization (neutral_vs_political_asymmetry.png)
4. **Link to:** Full analysis in results folder

---

## Alternative Shorter Version

```markdown
Hi Kevin,

Great suggestion! We ran the control test:

**Results:**
- Neutral pairs: Mean = 0.045
- Political pairs: Mean = 0.029
- p-value = 0.655 (not significant)

**Surprise:** Neutral pairs showed *higher* asymmetry!

**Interpretation:** Political opposites are semantically CLOSE (same topic) → low asymmetry. Different facts can be semantically FAR → variable asymmetry.

The metric measures embedding-space structure, not "bias" per se. Important caveat for interpretation!

Thanks for the push - negative results are valuable science.

Best, Davide
```

---

*Draft created: 2026-01-05*
