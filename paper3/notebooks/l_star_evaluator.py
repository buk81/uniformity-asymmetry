#!/usr/bin/env python3
"""
L* Formula Evaluator - Test new formulas against empirical data

Usage:
    python l_star_evaluator.py

Add your formula in the CANDIDATE_FORMULAS section.
"""

import numpy as np
from typing import Callable, Dict, List, Any

# =============================================================================
# COMPLETE DATASET
# =============================================================================

DATA = [
    {"model": "pythia-160m", "L": 12, "G": 1.157, "L_star": 7, "behavior": "DAMPEN", "n_heads": 12, "lab": "EleutherAI"},
    {"model": "pythia-410m", "L": 24, "G": 0.978, "L_star": 16, "behavior": "DAMPEN", "n_heads": 16, "lab": "EleutherAI"},
    {"model": "pythia-1b", "L": 16, "G": 1.216, "L_star": 15, "behavior": "DAMPEN", "n_heads": 8, "lab": "EleutherAI"},
    {"model": "pythia-2.8b", "L": 32, "G": 0.927, "L_star": 26, "behavior": "DAMPEN", "n_heads": 32, "lab": "EleutherAI"},
    {"model": "pythia-6.9b", "L": 32, "G": 0.994, "L_star": 21, "behavior": "DAMPEN", "n_heads": 32, "lab": "EleutherAI"},
    {"model": "opt-125m", "L": 12, "G": 1.263, "L_star": 8, "behavior": "EXPAND", "n_heads": 12, "lab": "Meta"},
    {"model": "gpt2", "L": 12, "G": 1.05, "L_star": 9, "behavior": "EXPAND", "n_heads": 12, "lab": "OpenAI"},
    {"model": "gemma-2b", "L": 18, "G": 1.0, "L_star": 17, "behavior": "UNKNOWN", "n_heads": 8, "lab": "Google"},
]

# W_V data for models with detailed analysis
WV_DATA = {
    "opt-125m": {"W_V_frob": 15.95, "W_V_spectral": 1.61, "entropy": 0.66},
    "gpt2": {"W_V_frob": 88.15, "W_V_spectral": 8.95, "entropy": 0.79},
    "pythia-160m": {"W_V_frob": 44.44, "W_V_spectral": 22.15, "entropy": 0.87},
}

# =============================================================================
# BASELINE FORMULAS (for comparison)
# =============================================================================

def formula_v1_original(d: Dict) -> float:
    """Original G-based formula. MAPE: 25.0%"""
    L, G = d["L"], d["G"]
    return (L / 2) * (1 + np.tanh(5 * (G - 1)))

def formula_simple_half(d: Dict) -> float:
    """Simple L/2 baseline. MAPE: 25.2%"""
    return d["L"] / 2

def formula_behavior_size(d: Dict) -> float:
    """Current best: Behavior + Size. MAPE: 10.0%"""
    L = d["L"]
    behavior = d["behavior"]
    if behavior == "DAMPEN":
        alpha, beta = 0.55, 0.008
    else:  # EXPAND or UNKNOWN
        alpha, beta = 0.60, 0.010
    return L * (alpha + beta * L)

def formula_linear_opt(d: Dict) -> float:
    """Optimized linear (behavior-agnostic). MAPE: 12.3%"""
    L = d["L"]
    return L * (0.544 + 0.0087 * L)

def formula_log_opt(d: Dict) -> float:
    """Optimized logarithmic. MAPE: 13.8%"""
    L = d["L"]
    return L * (0.31 + 0.17 * np.log(L))

# =============================================================================
# CANDIDATE FORMULAS - ADD YOUR PROPOSALS HERE
# =============================================================================

def formula_heads_aware(d: Dict) -> float:
    """
    Hypothesis: n_heads affects L* transition

    Observation: Models with n_heads=8 have late L* (~94%)
    """
    L = d["L"]
    n_heads = d["n_heads"]

    # Base ratio increases with depth
    base = 0.50 + 0.008 * L

    # Head correction: fewer heads → later transition
    head_factor = 1.0 + 0.02 * (12 - n_heads)  # 12 is reference

    return L * base * head_factor

def formula_gain_corrected(d: Dict) -> float:
    """
    Hypothesis: Use |G-1| instead of G-1 for symmetric response
    """
    L, G = d["L"], d["G"]

    base = 0.55 + 0.008 * L

    # Gain correction: deviation from unity matters
    gain_shift = 0.1 * (1 - abs(G - 1))

    return L * (base + gain_shift)

def formula_entropy_proxy(d: Dict) -> float:
    """
    Hypothesis: Attention entropy affects L*
    Use gain as entropy proxy (G~1 → higher entropy)
    """
    L, G = d["L"], d["G"]

    # Estimate entropy from gain (empirical relationship)
    entropy_proxy = 0.8 - 0.3 * abs(G - 1)

    # Higher entropy → later transition
    base = 0.50 + 0.15 * entropy_proxy
    depth_scale = 0.008 * L

    return L * (base + depth_scale)

def formula_power_law(d: Dict) -> float:
    """
    Hypothesis: L* follows power law scaling
    L* = a × L^b
    """
    L = d["L"]
    a = 0.45
    b = 1.08
    return a * (L ** b)

def formula_sigmoid_depth(d: Dict) -> float:
    """
    Hypothesis: L*/L ratio saturates at deep networks
    Uses sigmoid to model saturation
    """
    L = d["L"]

    # Sigmoid saturation: ratio → 0.85 as L → ∞
    max_ratio = 0.85
    min_ratio = 0.55
    k = 0.15  # steepness
    L_mid = 20  # midpoint

    ratio = min_ratio + (max_ratio - min_ratio) / (1 + np.exp(-k * (L - L_mid)))
    return L * ratio

# =============================================================================
# EVALUATION ENGINE
# =============================================================================

def evaluate_formula(formula: Callable, data: List[Dict], name: str) -> Dict[str, Any]:
    """Evaluate a formula on all data."""
    errors = []
    pct_errors = []
    predictions = []

    for d in data:
        pred = formula(d)
        emp = d["L_star"]
        err = abs(pred - emp)
        pct = err / d["L"] * 100

        errors.append(err)
        pct_errors.append(pct)
        predictions.append({
            "model": d["model"],
            "L": d["L"],
            "predicted": pred,
            "empirical": emp,
            "error": err,
            "pct_error": pct
        })

    return {
        "name": name,
        "mape": np.mean(pct_errors),
        "max_error": max(pct_errors),
        "predictions": predictions
    }

def print_results(result: Dict):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Formula: {result['name']}")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'L':>4} {'Pred':>7} {'Emp':>5} {'Err':>6} {'%Err':>8}")
    print("-"*50)

    for p in result["predictions"]:
        print(f"{p['model']:<15} {p['L']:>4} {p['predicted']:>7.1f} {p['empirical']:>5} {p['error']:>6.1f} {p['pct_error']:>7.1f}%")

    print("-"*50)
    print(f"MAPE: {result['mape']:.1f}%  |  Max Error: {result['max_error']:.1f}%")

def main():
    print("\n" + "="*70)
    print("L* FORMULA EVALUATOR")
    print("="*70)
    print(f"\nDataset: {len(DATA)} models")
    print("Target: MAPE < 5%")

    # All formulas to test
    formulas = [
        (formula_v1_original, "v1 (Original G-based)"),
        (formula_simple_half, "L/2 (Simple)"),
        (formula_behavior_size, "Behavior+Size (Current Best)"),
        (formula_linear_opt, "Linear (Optimized)"),
        (formula_log_opt, "Log (Optimized)"),
        # Candidates
        (formula_heads_aware, "Heads-Aware (NEW)"),
        (formula_gain_corrected, "Gain-Corrected (NEW)"),
        (formula_entropy_proxy, "Entropy-Proxy (NEW)"),
        (formula_power_law, "Power-Law (NEW)"),
        (formula_sigmoid_depth, "Sigmoid-Depth (NEW)"),
    ]

    # Evaluate all
    results = []
    for formula, name in formulas:
        result = evaluate_formula(formula, DATA, name)
        results.append(result)

    # Print detailed results for best formulas
    results.sort(key=lambda x: x["mape"])

    print("\n" + "="*70)
    print("LEADERBOARD")
    print("="*70)
    print(f"{'Rank':<6} {'Formula':<30} {'MAPE':>8} {'Max Err':>10} {'Status':<8}")
    print("-"*65)

    for i, r in enumerate(results, 1):
        status = "TARGET!" if r["mape"] < 5 else ("GOOD" if r["mape"] < 10 else "")
        print(f"{i:<6} {r['name']:<30} {r['mape']:>7.1f}% {r['max_error']:>9.1f}% {status:<8}")

    # Print details for top 3
    print("\n" + "="*70)
    print("TOP 3 DETAILED RESULTS")
    print("="*70)

    for r in results[:3]:
        print_results(r)

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS: Worst Predictions")
    print("="*70)

    best = results[0]
    worst_preds = sorted(best["predictions"], key=lambda x: -x["pct_error"])[:3]
    print(f"\nFormula: {best['name']}")
    print("Hardest models to predict:")
    for p in worst_preds:
        print(f"  - {p['model']}: {p['pct_error']:.1f}% error (pred={p['predicted']:.1f}, emp={p['empirical']})")

if __name__ == "__main__":
    main()
