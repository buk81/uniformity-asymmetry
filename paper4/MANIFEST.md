# Release Manifest

This document attests to the configuration and versions used for all experiments in this release package.

## Version Information

| Component | Version/Hash |
|-----------|--------------|
| **Repo Commit** | `07447cb20281f677cd00df8f6249e4e411ccfd66` |
| **Prompt Hash (SHA256)** | `25448fef934af177bb0fd9d933e58a949d40b9b5b2a48854bbcf8e366197b64a` |
| **Schema Version** | 1 |
| **Release Date** | 2026-01-20 |

## Software Versions (Tested Configuration)

| Package | Version |
|---------|---------|
| Python | 3.10.x |
| PyTorch | 2.1.0 |
| Transformers | 4.36.0 |
| Accelerate | 0.25.0 |
| BitsAndBytes | 0.41.0 |

## Hardware Used

- **Primary**: NVIDIA A100 80GB (Google Colab Pro+)
- **Validation**: NVIDIA RTX 4090 24GB (local)
- **8-bit Quantization**: Used for Gemma-2-27B and Falcon-40B on 40GB GPUs

## Artifact Counts

| Category | Count |
|----------|-------|
| A-Tier Notebooks | 19 |
| Result JSON Files | 24 |
| Supporting Evidence | 8 |
| **Total Artifacts** | **51** |

## Verification Checklist

- [ ] `python verify_release.py` passes
- [ ] All 19 A-tier notebooks present
- [ ] All result JSONs present
- [ ] Prompts SHA256 matches manifest
- [ ] Schema validation passes (with --check-schema)

## Reproducibility Notes

1. **Determinism**: Seeds [42, 123, 456] used for 3-run experiments
2. **Floating Point**: Results may vary ±1% due to GPU non-determinism
3. **Model Versions**: HuggingFace model revisions captured in result JSONs
4. **Quantization**: 8-bit results flagged in result metadata

## Attestation

This release package is frozen. All results correspond to the hashes listed above.

Any modifications to prompts, code, or methodology will be reflected in updated commit hashes.

---

*Generated: 2026-01-20*
*Author: Davide D'Elia*
