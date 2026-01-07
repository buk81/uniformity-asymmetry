# Artifact–Notebook Mapping (Paper 3)

This document maps each artifact (JSON/PNG) to the notebook that produces it.

**Status labels:** `core` = used in paper, `obsolete` = superseded, `rejected` = not used

---

## Core Artifacts

| Artifact | Notebook | Status | Notes |
|---|---|---|---|
| `thermodynamic_benchmark_*.json` | Grand_Unified_Thermodynamic_Benchmark | core | Fig. 2 |
| `grand_unified_benchmark_*.png` | Grand_Unified_Thermodynamic_Benchmark | core | 4-panel |
| `scaling_law_multi_pythia_*.json` | Scaling_Law_Multi_Pythia | core | Fig. 1 |
| `scaling_law_multi_pythia.png` | Scaling_Law_Multi_Pythia | core | 4-panel |
| `restriction_maps_results.json` | Restriction_Maps_Extraction | core | Fig. A5 |
| `sheaf_laplacian_spectral.png` | Restriction_Maps_Extraction | core | |
| `twin_test_results_*.json` | Twin_Test_Base_vs_Instruct | core | §5.5 |
| `twin_test_results_*.png` | Twin_Test_Base_vs_Instruct | core | 4-panel |
| `4model_validation_results.json` | 4Model_Cross_Architecture_Validation | core | |
| `4model_validation_plot.png` | 4Model_Cross_Architecture_Validation | core | |
| `4model_cumulative_energy.png` | 4Model_Cross_Architecture_Validation | core | |
| `mistral_paradox_investigation_*.json` | Mistral_Paradox_Investigation | core | §5.6 |
| `OPT_anomaly_*.json` | OPT_Anomaly_Investigation | core | Appendix A |
| `gpt2_layernorm_validation_*.json` | GPT2_LayerNorm_Validation | core | |
| `gptj_parallel_test_*.json` | GPTJ_Parallel_Architecture_Test | core | |
| `l_star_sign_change_*.json` | L_Star_Cross_Heritage_SignChange | core | |
| `H4_validation_*.json` | H4_v2_Extended_Models | core | |
| `H4_v2_validation.png` | H4_v2_Extended_Models | core | 6-panel |

---

## Historical Notes (removed from repo)

The following artifacts were generated during development but removed as obsolete:
- `high_rho_model_hunt_*.json` - LayerNorm artifact, superseded by NO_FINAL_LN
- `pythia_family_residual_gain_*.json` - Superseded by NO_FINAL_LN version
- `rlhf_safety_brake_test_results.json` - Hypothesis rejected, not used in paper

---

## Final Validated Artifacts

| Artifact | Notebook | Notes |
|---|---|---|
| `high_rho_hunt_NO_FINAL_LN_*.json` | High_Rho_Model_Hunt_NO_FINAL_LN | Final H26 |
| `clean_residual_NO_FINAL_LN_*.json` | Clean_Residual_Gain_NO_FINAL_LN | Clean metric proof |
| `pythia_family_NO_FINAL_LN_*.json` | Pythia_Family_Residual_Gain_Sweep | Final H25 |

---

## Notebook → Artifact Quick Reference

| Notebook | Outputs |
|---|---|
| Scaling_Law_Multi_Pythia | `scaling_law_multi_pythia_*.{json,png}` |
| Grand_Unified_Thermodynamic_Benchmark | `thermodynamic_benchmark_*.json`, `grand_unified_benchmark_*.png` |
| Restriction_Maps_Extraction | `restriction_maps_results.json`, `sheaf_laplacian_spectral.png` |
| Twin_Test_Base_vs_Instruct | `twin_test_results_*.{json,png}` |
| 4Model_Cross_Architecture_Validation | `4model_*.{json,png}` |
| Mistral_Paradox_Investigation | `mistral_paradox_*.{json,png}` |
| OPT_Anomaly_Investigation | `OPT_anomaly_*.{json,png}` |
| GPT2_LayerNorm_Validation | `gpt2_layernorm_*.{json,png}` |
| GPTJ_Parallel_Architecture_Test | `gptj_parallel_test_*.{json,png}` |
| L_Star_Cross_Heritage_SignChange | `l_star_sign_change_*.{json,png}` |
| H4_v2_Extended_Models | `H4_*.{json,csv,png}` |
| Hypothesis_Tests_LLaMA_Anomaly | `llama_anomaly_*.json` |
| FFN_Expansion_Analysis | `ffn_expansion_*.{json,png}` |
| Bentov_Point_Characterization | `BENTOV_POINT_DISCOVERY.md` (theoretical framework) |
| Clean_Residual_Gain_NO_FINAL_LN | `clean_residual_NO_FINAL_LN_*.json` (Fig. A1) |
| High_Rho_Model_Hunt_NO_FINAL_LN | `high_rho_hunt_NO_FINAL_LN_*.json` (§5.4) |
| Pythia_Family_Residual_Gain_Sweep | `pythia_family_NO_FINAL_LN_*.json` (Fig. A3) |
| Input_Dependency_Thermodynamics | `input_dependency_thermodynamics.json` (Fig. A4) |

---

## Reproducibility

All experiments use `PYTHONHASHSEED=42` for deterministic results.
