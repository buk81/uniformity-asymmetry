# Artifact–Notebook Mapping (Paper 3 Core Set)

Quelle: PAPER_3_PROTOCOL.md (Kanon). Dieses Dokument listet alle explizit referenzierten Artefakte (JSON/PNG) und das Notebook/Script, das sie erzeugt. Status nach Protocol: **core**, **obsolet**, **rejected**.

## Kern-Artefakte

| Artifact | Notebook / Script | Status | Hinweis |
|---|---|---|---|
| H4_validation_20260106_014132.json | H4_validation_local.py / H4_v2_Extended_Models.ipynb | core | 4 Modelle, 4 Prompts |
| H4_v2_validation_20260106_015547.json | H4_validation_local.py / H4_v2_Extended_Models.ipynb | core | Figure: H4_v2_validation.png |
| H4_v2_validation.png | H4_v2_Extended_Models.ipynb | core | 6-Panel |
| restriction_maps_results.json | Restriction_Maps_Extraction.ipynb | core | |
| restriction_maps_analysis.png | Restriction_Maps_Extraction.ipynb | core | |
| sheaf_laplacian_spectral.png | Restriction_Maps_Extraction.ipynb | core | |
| thermodynamic_benchmark_20260105_141034.json | Grand_Unified_Thermodynamic_Benchmark.ipynb | core | "Grand Unified" |
| grand_unified_benchmark_20260105_141034.png | Grand_Unified_Thermodynamic_Benchmark.ipynb | core | 4-Panel |
| anisotropy_results_pythia.json | Anisotropy_Profile_Pythia.ipynb | core | |
| anisotropy_results_gemma.json | Anisotropy_Profile_Gemma.ipynb | core | |
| anisotropy_profile_pythia.png | Anisotropy_Profile_Pythia.ipynb | core | |
| anisotropy_profile_gemma.png | Anisotropy_Profile_Gemma.ipynb | core | |
| ffn_expansion_results.json | FFN_Expansion_Analysis.ipynb | core | 1.4B |
| ffn_expansion_pythia69b_results.json | FFN_Expansion_Pythia6.9B.ipynb | core | 6.9B |
| ffn_expansion_analysis.png | FFN_Expansion_Analysis.ipynb | core | |
| ffn_expansion_pythia69b_analysis.png | FFN_Expansion_Pythia6.9B.ipynb | core | |
| scaling_law_multi_pythia_results.json | Scaling_Law_Multi_Pythia.ipynb | core | |
| scaling_law_multi_pythia.png | Scaling_Law_Multi_Pythia.ipynb | core | 4-Panel |
| cross_architecture_validation_results.json | Cross_Architecture_Validation.ipynb / 4Model_Cross_Architecture_Validation.ipynb | core | |
| cross_architecture_validation.png | Cross_Architecture_Validation.ipynb | core | |
| 4model_validation_results.json | 4Model_Cross_Architecture_Validation.ipynb | core | |
| 4model_validation_plot.png | 4Model_Cross_Architecture_Validation.ipynb | core | |
| 4model_cumulative_energy.png | 4Model_Cross_Architecture_Validation.ipynb | core | |
| llama_anomaly_hypothesis_tests.json | Hypothesis_Tests_LLaMA_Anomaly.ipynb | core | Titanium Projector + Long-Context |
| llama2_vs_llama31_hypothesis_test.json | Hypothesis_Tests_LLaMA_Anomaly.ipynb | core | inverted results ⚠️ |
| mistral_paradox_investigation_results.json | Mistral_Paradox_Investigation.ipynb | core | |
| mistral_paradox_investigation.png | Mistral_Paradox_Investigation.ipynb | core | |
| OPT_anomaly_20260106_013155.json | OPT_Anomaly_Investigation.ipynb | core | Appendix A |
| OPT_anomaly_investigation.png | OPT_Anomaly_Investigation.ipynb | core | Appendix A |
| gpt2_layernorm_validation_20260105_150306.json | GPT2_LayerNorm_Validation.ipynb | core | |
| gpt2_layernorm_validation_20260105_150306.png | GPT2_LayerNorm_Validation.ipynb | core | |
| gpt2_vs_reference_20260105_150306.png | GPT2_LayerNorm_Validation.ipynb | core | |
| gptj_parallel_test_20260105_152333.json | GPTJ_Parallel_Architecture_Test.ipynb | core | |
| gptj_parallel_test_20260105_152333.png | GPTJ_Parallel_Architecture_Test.ipynb | core | |
| twin_test_results_20260105_160237.json | Twin_Test_Base_vs_Instruct.ipynb | core | 150 measurements |
| twin_test_results_20260105_160237.png | Twin_Test_Base_vs_Instruct.ipynb | core | 4-Panel |
| beautiful_ones_analysis_20260105_164639.json | Beautiful_Ones_Per_Head_Analysis.ipynb | core | |
| beautiful_ones_heatmap.png | Beautiful_Ones_Per_Head_Analysis.ipynb | core | |
| residual_growth_profile.png | Beautiful_Ones_Per_Head_Analysis.ipynb | core | |
| l_star_sign_change_20260106_111315.json | L_Star_Cross_Heritage_SignChange.ipynb | core | |
| l_star_sign_change_20260106_111315.png | L_Star_Cross_Heritage_SignChange.ipynb | core | |
| bentov_point_* (BENTOV_POINT_DISCOVERY / Gain) | Bentov_Point_Characterization.ipynb | core | Kernthese Bentov Point |

## Weitere erwähnte / spezielle Status

| Artifact | Status | Hinweis |
|---|---|---|
| rlhf_safety_brake_test_results.json | rejected | Protocol markiert als ❌ |
| high_rho_model_hunt_20260105_170840.json | obsolet | LN-Artefakt laut Protocol |
| pythia_family_residual_gain_20260105_173135.json | obsolet | ALT, Artefakt |
| high_rho_hunt_NO_FINAL_LN_20260105_184728.json | core (Final) | FINALE H26 |
| clean_residual_NO_FINAL_LN_20260105_173546.json | core (Final) | Clean Metrik Beweis |
| pythia_family_NO_FINAL_LN_20260105_174820.json | core (Final) | FINALE H25 |

## Notebooks → Artefakte (Kurzliste)
- Anisotropy_Profile_Gemma.ipynb → anisotropy_results_gemma.json/.png
- Anisotropy_Profile_Pythia.ipynb → anisotropy_results_pythia.json/.png
- 4Model_Cross_Architecture_Validation.ipynb → 4model_validation_results.json, 4model_validation_plot.png, 4model_cumulative_energy.png, residual_stream_complete_validation.png
- Cross_Architecture_Validation.ipynb → cross_architecture_validation_results.json, cross_architecture_validation.png
- FFN_Expansion_Analysis.ipynb → ffn_expansion_results.json, ffn_expansion_analysis.png
- FFN_Expansion_Pythia6.9B.ipynb → ffn_expansion_pythia69b_results.json, ffn_expansion_pythia69b_analysis.png
- Restriction_Maps_Extraction.ipynb → restriction_maps_results.json, restriction_maps_analysis.png, sheaf_laplacian_spectral.png
- Grand_Unified_Thermodynamic_Benchmark.ipynb → thermodynamic_benchmark_20260105_141034.json, grand_unified_benchmark_20260105_141034.png
- Hypothesis_Tests_LLaMA_Anomaly.ipynb → llama_anomaly_hypothesis_tests.json, llama2_vs_llama31_hypothesis_test.json, titanium_projector_hypothesis.png
- Mistral_Paradox_Investigation.ipynb → mistral_paradox_investigation_results.json/.png
- OPT_Anomaly_Investigation.ipynb → OPT_anomaly_20260106_013155.json/.png
- GPT2_LayerNorm_Validation.ipynb → gpt2_layernorm_validation_20260105_150306.{json,png}, gpt2_vs_reference_20260105_150306.png
- GPTJ_Parallel_Architecture_Test.ipynb → gptj_parallel_test_20260105_152333.{json,png}
- Scaling_Law_Multi_Pythia.ipynb → scaling_law_multi_pythia_results.json, scaling_law_multi_pythia.png
- Twin_Test_Base_vs_Instruct.ipynb → twin_test_results_20260105_160237.{json,png}
- Beautiful_Ones_Per_Head_Analysis.ipynb → beautiful_ones_analysis_20260105_164639.json, beautiful_ones_heatmap.png, residual_growth_profile.png
- L_Star_Cross_Heritage_SignChange.ipynb → l_star_sign_change_20260106_111315.{json,png}
- Bentov_Point_Characterization.ipynb → bentov_point_* (Gain/Bentov Point Plots)
- Clean_Residual_Gain_NO_FINAL_LN.ipynb → clean_residual_NO_FINAL_LN_20260105_173546.{json,png}
- Pythia_Family_Residual_Gain_Sweep.ipynb / High_Rho_Model_Hunt(_NO_FINAL_LN).ipynb → pythia_family_NO_FINAL_LN_20260105_174820.json, high_rho_hunt_NO_FINAL_LN_20260105_184728.json



