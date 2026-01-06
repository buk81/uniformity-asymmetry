#!/usr/bin/env python3
"""
L* Formula Improvement Analysis

Current formula: L* ≈ (L/2) × (1 + tanh(κ(G-1)))
Goal: Reduce 25% error to <15%

Uses combined data from H4 v2 and Extended Models experiments.
"""

import numpy as np
import json

# =============================================================================
# DATA FROM EXPERIMENTS
# =============================================================================

# H4 v2 Original (4 models)
h4_v2_data = [
    {"model": "pythia-160m", "L": 12, "G": 1.157, "L_star_apriori": 9.9, "L_star_emp": 7, "behavior": "DAMPEN"},
    {"model": "pythia-410m", "L": 24, "G": 0.978, "L_star_apriori": 10.7, "L_star_emp": 16, "behavior": "DAMPEN"},
    {"model": "opt-125m", "L": 12, "G": 1.263, "L_star_apriori": 11.2, "L_star_emp": 8, "behavior": "EXPAND"},
    {"model": "gpt2", "L": 12, "G": 1.05, "L_star_apriori": 7.5, "L_star_emp": 9, "behavior": "EXPAND"},
]

# Extended Models (5 models) - from H4_v2_extended
extended_data = [
    {"model": "pythia-1b", "L": 16, "G": 1.216, "L_star_emp": 15, "behavior": "DAMPEN"},
    {"model": "pythia-2.8b", "L": 32, "G": 0.927, "L_star_emp": 26, "behavior": "DAMPEN"},
    {"model": "pythia-6.9b", "L": 32, "G": 0.994, "L_star_emp": 21, "behavior": "DAMPEN"},
    {"model": "Mistral-7B", "L": 32, "G": 1.05, "L_star_emp": 0, "behavior": "EXPAND"},  # Anomalous!
    {"model": "Gemma-2b", "L": 18, "G": 1.0, "L_star_emp": 17, "behavior": "UNKNOWN"},
]

# OPT Investigation data - W_V norms and entropy
wv_data = {
    "opt-125m": {"W_V_frob": 15.95, "entropy": 0.66},
    "gpt2": {"W_V_frob": 88.15, "entropy": 0.79},
    "pythia-160m": {"W_V_frob": 44.44, "entropy": 0.87},
}

# =============================================================================
# FORMULA CANDIDATES
# =============================================================================

def l_star_v1(L, G, kappa=5.0):
    """Original formula from paper."""
    return (L / 2) * (1 + np.tanh(kappa * (G - 1)))

def l_star_simple(L, G):
    """Simple L/2 baseline."""
    return L / 2

def l_star_v2(L, G, alpha=0.3):
    """Improved formula v2: L/2 with gain modulation."""
    # Key insight: For DAMPEN (G<1), L* > L/2
    # For EXPAND (G>1), L* can be < or > L/2
    base = L / 2
    # Shift based on gain
    if G < 1:  # DAMPEN
        shift = alpha * L * (1 - G)  # Positive shift
    else:  # EXPAND
        shift = -alpha * L * (G - 1) * 0.5  # Smaller negative shift
    return base + shift

def l_star_v3(L, G, behavior):
    """Behavior-aware formula v3."""
    base = L / 2
    if behavior == "DAMPEN":
        # DAMPEN models: L* tends toward 2L/3
        return base + 0.15 * L
    else:  # EXPAND or UNKNOWN
        # EXPAND models: L* closer to L/2 or slightly higher
        return base + 0.1 * L * (G - 1)

def l_star_v4(L, G):
    """
    Final improved formula v4:

    L* = (L/2) × (1 + β × sign(1-G) × |1-G|^0.5)

    - Uses sqrt of deviation for softer response
    - sign() captures direction
    - Works for both DAMPEN and EXPAND
    """
    beta = 0.8
    deviation = 1 - G
    sign = np.sign(deviation)
    magnitude = np.sqrt(np.abs(deviation))
    return (L / 2) * (1 + beta * sign * magnitude)

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_formula(formula, data, name):
    """Evaluate a formula on data."""
    errors = []
    pct_errors = []

    print(f"\n{'='*60}")
    print(f"Formula: {name}")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'L':>4} {'G':>6} {'L*_pred':>8} {'L*_emp':>8} {'Error':>8} {'%Err':>8}")
    print("-"*60)

    for d in data:
        L = d["L"]
        G = d["G"]
        L_star_emp = d["L_star_emp"]

        # Skip anomalous Mistral (L*=0 is clearly wrong)
        if L_star_emp == 0:
            print(f"{d['model']:<15} {L:>4} {G:>6.3f} {'SKIP':>8} {L_star_emp:>8} {'N/A':>8} {'N/A':>8}")
            continue

        if "behavior" in d:
            if name == "v3 (Behavior)":
                L_star_pred = formula(L, G, d["behavior"])
            else:
                L_star_pred = formula(L, G)
        else:
            L_star_pred = formula(L, G)

        error = abs(L_star_pred - L_star_emp)
        pct_error = error / L * 100

        errors.append(error)
        pct_errors.append(pct_error)

        print(f"{d['model']:<15} {L:>4} {G:>6.3f} {L_star_pred:>8.1f} {L_star_emp:>8} {error:>8.1f} {pct_error:>7.1f}%")

    mean_error = np.mean(errors)
    mean_pct_error = np.mean(pct_errors)

    print("-"*60)
    print(f"{'MEAN':<15} {'':<4} {'':<6} {'':<8} {'':<8} {mean_error:>8.1f} {mean_pct_error:>7.1f}%")

    return mean_error, mean_pct_error

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("L* FORMULA IMPROVEMENT ANALYSIS")
    print("="*70)
    print("\nGoal: Reduce error from 25% to <15%")

    # Combine all data (excluding Mistral anomaly for fitting)
    all_data = h4_v2_data + extended_data

    # Evaluate each formula
    results = {}

    # v1 Original (with kappa=5)
    results["v1 (Original)"] = evaluate_formula(
        lambda L, G: l_star_v1(L, G, kappa=5.0),
        all_data, "v1 (Original)"
    )

    # Simple L/2
    results["L/2 (Simple)"] = evaluate_formula(
        l_star_simple,
        all_data, "L/2 (Simple)"
    )

    # v2 Gain-modulated
    results["v2 (Gain-mod)"] = evaluate_formula(
        l_star_v2,
        all_data, "v2 (Gain-mod)"
    )

    # v4 Sqrt-based
    results["v4 (Sqrt)"] = evaluate_formula(
        l_star_v4,
        all_data, "v4 (Sqrt)"
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Formula Comparison")
    print("="*70)
    print(f"{'Formula':<20} {'Mean Error':>12} {'Mean % Error':>14}")
    print("-"*50)

    for name, (err, pct) in sorted(results.items(), key=lambda x: x[1][1]):
        status = "✓" if pct < 15 else "✗"
        print(f"{name:<20} {err:>10.2f} L {pct:>12.1f}% {status}")

    # Best formula
    best = min(results.items(), key=lambda x: x[1][1])
    print("\n" + "="*70)
    print(f"BEST FORMULA: {best[0]}")
    print(f"Mean Error: {best[1][0]:.2f} layers ({best[1][1]:.1f}%)")
    print("="*70)

    # Final recommendation
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    print("""
If best is v4 (Sqrt):
    L* = (L/2) × (1 + 0.8 × sign(1-G) × √|1-G|)

Key insight: The sqrt dampens extreme deviations,
making predictions more robust across model families.

If best is L/2 (Simple):
    L* ≈ L/2

Key insight: The transition point is remarkably close to
the midpoint for most architectures. Additional factors
(entropy, W_V) may matter for fine-tuning but L/2 is a
strong baseline.
""")
