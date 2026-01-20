# Paper 4: Alignment Robustness - Reproducibility Package

> "Architecture determines alignment fragility more than scale or training data."

This package contains all notebooks, results, and tooling required to reproduce the experiments in Paper 4.

## Quick Start

```bash
# 1. Verify package integrity
python verify_release.py

# 2. Reproduce E11 experiment (requires A100 GPU)
python reproduce.py --experiment E11 --model mistral --seed 42

# 3. Or use notebooks in Colab (A100 recommended)
# Upload any notebook from notebooks/ to Google Colab
```

## Naming Convention

| Symbol | Meaning | Example |
|--------|---------|---------|
| **A#** | Claim (narrative unit) | A1 = Territorial Collapse |
| **E##** | Experiment (measurement protocol) | E11 = Perturbation Probe |
| **Artifact** | Notebook + Result JSON | E11_*.ipynb + E11_*.json |

Each claim maps to one or more experiments. See `claims.yaml` for machine-readable mapping.

## A-Tier Claims

| ID | Claim | Notebooks | Key Metric |
|----|-------|-----------|-----------|
| A1 | Territorial Collapse | `A1_territorial/` (6) | ΔSI varies by architecture |
| A2 | Indra State-Dependency | `A2_indra/` (6) | +114% heal / -7.8% damage |
| A3 | Heritage > Scale | `A3_heritage/` (3) | RLHF fragility pattern |
| A4 | Synthetic Immunity | `A4_synthetic/` (3) | Phi SI ≈ 0.33 invariant |
| A5 | MQA Pre-Collapsed | `A5_mqa/` (1) | Falcon SI ≈ 0.14 |

## Key Findings

- **GQA Noise Amplification**: ~5,800× PPL-slope ratio (E06b)
- **Same-Family Amplification**: ~180,000× (E06d)
- **Dimensional Crowding**: 65,799× GQA/MHA ratio (E21)
- **Statistical Validation**: Bootstrap CI, all A-tier p < 0.01

## Methodology Standard (E11-v3)

| Parameter | Value |
|-----------|-------|
| Seeds | [42, 123, 456] |
| Prompts | `prompts/standard10_v3.txt` |
| Prompt Hash (SHA256) | `25448fef934af177bb0fd9d933e58a949d40b9b5b2a48854bbcf8e366197b64a` |
| MAX_LENGTH | 128 |
| dtype | torch.bfloat16 |

## Directory Structure

```
github_release/
├── README.md                    # This file
├── claims.yaml                  # Machine-readable claim mapping
├── reproduce.py                 # CLI reproduction tool
├── verify_release.py            # Package integrity checker
├── requirements.txt             # Pinned dependencies
├── requirements-min.txt         # Minimum dependencies
│
├── prompts/
│   └── standard10_v3.txt        # Standard-10 v3 prompts
│
├── notebooks/
│   ├── A1_territorial/          # 6 notebooks
│   ├── A2_indra/                # 6 notebooks
│   ├── A3_heritage/             # 3 notebooks
│   ├── A4_synthetic/            # 3 notebooks
│   ├── A5_mqa/                  # 1 notebook
│   └── supporting/              # B-tier evidence
│
├── results/
│   ├── schema_v1.json           # Result JSON schema
│   ├── E11/                     # Territorial + Indra results
│   ├── E04/                     # Heritage results
│   ├── E06/                     # Triangle/Same-Scale/Same-Family
│   ├── E08/                     # Phi synthetic immunity
│   ├── E21_E22/                 # Dimensional crowding, Indra Cure
│   └── E-ISO/                   # Isomorphism validation
│
└── src/
    └── metrics.py               # Shared SI/PPL computation
```

## Reproduction Options

### Option 1: CLI (recommended for verification)

```bash
pip install -r requirements.txt
python reproduce.py --experiment E11 --model mistral --seed 42
```

**Note:** CLI currently supports E11 (territorial) and E04 (heritage) minimal reproduction; notebooks remain canonical for full sweep.

### Option 2: Colab Notebooks

1. Upload notebook to Google Colab
2. Select A100 GPU runtime (required for 7B+ models)
3. Execute Cell 0b first (HF login, GPU check)
4. Run All
5. Compare output with `results/` JSON

### Option 3: Local Execution

```bash
# With GPU
pip install -r requirements.txt
jupyter notebook notebooks/A1_territorial/E11_Territorial_Collapse.ipynb
```

## File Verification

```bash
# Basic verification
python verify_release.py

# With schema validation
python verify_release.py --check-schema

# Verbose output
python verify_release.py --verbose
```

## Model Requirements

| Model | VRAM | Architecture |
|-------|------|--------------|
| Mistral-7B | 16GB | GQA-8 |
| LLaMA-2-7B | 16GB | MHA-32 |
| LLaMA-3.1-8B | 18GB | GQA-8 |
| Gemma-2-9B | 20GB | GQA-4 |
| Gemma-2-27B | 55GB (8-bit: 32GB) | GQA-8 |
| Falcon-40B | 80GB (8-bit: 45GB) | MQA-1 |

## Citation

```bibtex
@misc{delia2026alignment,
  title={Alignment Robustness: How Attention Architecture Determines RLHF Fragility},
  author={D'Elia, Davide},
  year={2026},
  howpublished={GitHub},
  url={https://github.com/buk81/uniformity-asymmetry/tree/main/paper4}
}
```

## License

MIT License - See repository root for details.

## Contact

For questions about reproduction or methodology, please open an issue in the main repository.
