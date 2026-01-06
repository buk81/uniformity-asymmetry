# LaTeX Submission Package

## Files

```
latex/
├── thermodynamic_constraints.tex   # Main document (v3.8 FINAL)
├── references.bib                  # BibTeX references
├── README.md                       # This file
└── neurips_2024.sty               # Download from venue (see below)
```

## Setup

1. **Download style file** from your target venue:
   - NeurIPS 2024: https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles
   - ICLR 2025: https://github.com/ICLR/Master-Template

2. **Place style file** in this directory

3. **Compile**:
   ```bash
   pdflatex thermodynamic_constraints.tex
   bibtex thermodynamic_constraints
   pdflatex thermodynamic_constraints.tex
   pdflatex thermodynamic_constraints.tex
   ```

## Switching Venues

In `thermodynamic_constraints.tex`, uncomment the appropriate line:

```latex
% \usepackage{neurips_2024}        % NeurIPS 2024
% \usepackage{iclr2025_conference} % ICLR 2025
\usepackage[preprint]{neurips_2024} % Preprint mode (current)
```

## Figure Paths

Figures are referenced from `../Figures/`. Ensure the relative path is correct or copy figures to `latex/Figures/`.

## Checklist Before Submission

- [ ] Remove `[preprint]` option for anonymous submission
- [ ] Verify page limit (NeurIPS: 9 pages main, unlimited appendix)
- [ ] Check figure resolution (300 DPI minimum)
- [ ] Run `pdffonts` to verify all fonts embedded
- [ ] Validate PDF/A compliance if required

## Version

v3.8 FINAL (2026-01-06)
