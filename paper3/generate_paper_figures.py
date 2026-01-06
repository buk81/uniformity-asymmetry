#!/usr/bin/env python3
"""
Generate all figures for Paper #3: Thermodynamic Constraints in Transformer Architectures

Figures:
1. Kleiber's Law - G_max vs 1/L scaling
2. Training Heritage - Lab signature bar chart
3. Spectral Signature - W_V/W_O scatter by lab
4. Layer-wise Dynamics - Expansion/dampening patterns
5. Lab Signature Comparison - Mean gain by lab

Author: Davide D'Elia
Date: 2026-01-05
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Color scheme (colorblind-friendly)
COLORS = {
    'EleutherAI': '#E74C3C',  # Red
    'Meta': '#3498DB',         # Blue
    'BigScience': '#27AE60',   # Green
    'OpenAI': '#9B59B6',       # Purple
    'TII': '#F39C12',          # Orange
    'StabilityAI': '#1ABC9C',  # Teal
}

def load_results():
    """Load all result files."""
    results_dir = Path(__file__).parent / "Results"

    with open(results_dir / "pythia_family_NO_FINAL_LN_20260105_174820.json") as f:
        pythia_data = json.load(f)

    with open(results_dir / "high_rho_hunt_NO_FINAL_LN_20260105_184728.json") as f:
        high_rho_data = json.load(f)

    with open(results_dir / "restriction_map_spectral_20260105_202448.json") as f:
        spectral_data = json.load(f)

    return pythia_data, high_rho_data, spectral_data


def figure1_kleiber_law(pythia_data, save_path):
    """
    Figure 1: Kleiber's Law for Transformers
    G_max scales as 10^(1/L) with depth L
    Shows r = -0.878 correlation between rho and gain
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Extract data
    models = pythia_data['models']
    n_layers = [m['n_layers'] for m in models]
    gains = [m['residual_gain_mean'] for m in models]
    rho = [m['rho'] for m in models]
    is_dampening = [m['is_dampening'] for m in models]

    # Calculate inverse layers
    inv_L = [1/L for L in n_layers]

    # Theoretical Kleiber line: G_max = 10^(1/L)
    L_range = np.linspace(6, 40, 100)
    G_theoretical = 10 ** (1/L_range)

    # Plot theoretical line
    ax.plot(1/L_range, G_theoretical, 'k--', alpha=0.5, linewidth=2,
            label=r'Kleiber Bound: $G_{max} = 10^{1/L}$')

    # Custom label offsets per model (x_offset, y_offset in points)
    # STRATEGY: Labels on opposite sides for overlapping x-coords
    #
    # Data positions (sorted by x):
    #   12b:  x=0.028, y=0.986 (leftmost) --> label LEFT
    #   2.8b: x=0.031, y=0.927 (lowest)   --> label LEFT-DOWN
    #   6.9b: x=0.031, y=0.994 (same x!)  --> label RIGHT-UP (far!)
    #   410m: x=0.042, y=0.978            --> label LEFT-DOWN
    #   1.4b: x=0.042, y=1.005 (same x!)  --> label RIGHT-UP (far!)
    #   1b:   x=0.063, y=1.216 (highest)  --> label LEFT
    #   160m: x=0.083, y=1.157            --> label RIGHT
    #   70m:  x=0.167, y=1.201 (rightmost)--> label RIGHT-DOWN
    label_offsets = {
        'pythia-70m': (25, -30),       # Right-down (space available)
        'pythia-160m': (30, 25),       # Right-up
        'pythia-1b': (-55, 20),        # Left-up (avoid Kleiber line)
        'pythia-410m': (-70, -20),     # LEFT-down (opposite of 1.4b)
        'pythia-1.4b': (65, 55),       # RIGHT-up VERY HIGH (avoid 6.9b!)
        'pythia-2.8b': (-75, -5),      # LEFT (isolated at bottom)
        'pythia-6.9b': (75, 20),       # RIGHT, moderate height (below 1.4b)
        'pythia-12b': (-70, 25),       # LEFT-up (above 410m)
    }

    # First pass: plot all points
    dampener_plotted = False
    expander_plotted = False

    for i, m in enumerate(models):
        color = COLORS['EleutherAI'] if is_dampening[i] else '#3498DB'
        marker = 'v' if is_dampening[i] else '^'

        # Add legend label only once per type
        if is_dampening[i] and not dampener_plotted:
            label = 'Dampener (G < 1)'
            dampener_plotted = True
        elif not is_dampening[i] and not expander_plotted:
            label = 'Expander (G > 1)'
            expander_plotted = True
        else:
            label = None

        ax.scatter(inv_L[i], gains[i], c=color, marker=marker, s=180,
                  edgecolor='black', linewidth=1.5, label=label, zorder=5)

    # Second pass: add labels with arrows
    for i, m in enumerate(models):
        model_name = m['model']
        x_off, y_off = label_offsets.get(model_name, (25, 0))
        short_name = model_name.replace('pythia-', '')
        color = COLORS['EleutherAI'] if is_dampening[i] else '#3498DB'

        ax.annotate(
            short_name,
            xy=(inv_L[i], gains[i]),
            xytext=(x_off, y_off),
            textcoords="offset points",
            fontsize=10,
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=color, linewidth=1.5, alpha=0.95),
            arrowprops=dict(
                arrowstyle='-',
                color=color,
                linewidth=1.2,
                shrinkA=0,
                shrinkB=5
            ),
            zorder=10
        )

    # Horizontal line at G=1
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=2,
               alpha=0.7, label='Neutral (G = 1)')

    # Add correlation annotation (top-left, out of the way)
    r = pythia_data['analysis']['rho_gain_correlation']
    p = pythia_data['analysis']['correlation_p_value']
    ax.text(0.02, 0.02, f'Correlation: $r$ = {r:.3f}, $p$ = {p:.4f}',
           transform=ax.transAxes, fontsize=11,
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    ax.set_xlabel('Inverse Depth (1/L)', fontsize=12)
    ax.set_ylabel('Mean Residual Gain', fontsize=12)
    ax.set_title("Figure 1: Kleiber's Law for Transformers\n"
                 "Deeper Networks Require Dampening for Stability",
                 fontsize=13, fontweight='bold')

    # Legend in lower-right corner (no data there)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.20)
    ax.set_ylim(0.88, 1.28)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def figure2_training_heritage(high_rho_data, save_path):
    """
    Figure 2: Training Heritage Dominance
    Bar chart showing dampening percentage by lab
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    lab_data = high_rho_data['h26_lab_signatures']

    # Sort by dampening percentage
    labs = sorted(lab_data.keys(), key=lambda x: lab_data[x]['dampen_pct'], reverse=True)

    dampen_pcts = [lab_data[lab]['dampen_pct'] for lab in labs]
    expand_pcts = [100 - d for d in dampen_pcts]
    n_models = [int(lab_data[lab]['n_models']) for lab in labs]

    x = np.arange(len(labs))
    width = 0.35

    # Stacked bar
    bars1 = ax.bar(x, dampen_pcts, width, label='Dampening Models',
                   color=COLORS['EleutherAI'], edgecolor='black')
    bars2 = ax.bar(x, expand_pcts, width, bottom=dampen_pcts, label='Expanding Models',
                   color=COLORS['Meta'], edgecolor='black')

    # Add count annotations
    for i, (lab, n) in enumerate(zip(labs, n_models)):
        ax.annotate(f'n={n}', (i, 105), ha='center', fontsize=10)

    # Add percentage labels
    for i, (d, e) in enumerate(zip(dampen_pcts, expand_pcts)):
        if d > 0:
            ax.annotate(f'{d:.0f}%', (i, d/2), ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')
        if e > 0:
            ax.annotate(f'{e:.0f}%', (i, d + e/2), ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')

    ax.set_xlabel('Research Lab')
    ax.set_ylabel('Percentage of Models')
    ax.set_title('Figure 2: Training Heritage Dominates Architecture\nDampening vs Expanding Models by Lab')
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=0)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 115)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    # Color the x-tick labels by signature
    for i, lab in enumerate(labs):
        color = COLORS.get(lab, 'black')
        ax.get_xticklabels()[i].set_color(color)
        ax.get_xticklabels()[i].set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def figure3_spectral_signature(spectral_data, save_path):
    """
    Figure 3: Spectral Signature Correspondence
    W_V vs W_O scatter showing 10x difference between labs
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for result in spectral_data['results']:
        lab = result['lab']
        model_name = result['model'].split('/')[-1]
        W_V = result['W_V_mean_spectral']
        W_O = result['W_O_mean_spectral']

        color = COLORS.get(lab, 'gray')
        marker = 'o' if result['expected'] == 'DAMPENER' else 's' if result['expected'] == 'EXPANDER' else '^'

        ax.scatter(W_V, W_O, c=color, marker=marker, s=200,
                  edgecolor='black', linewidth=1.5, label=f'{lab}: {model_name}',
                  alpha=0.8)

        # Annotate
        ax.annotate(model_name, (W_V, W_O), textcoords="offset points",
                   xytext=(8, 5), fontsize=9, alpha=0.8)

    # Add regions
    ax.axvspan(0, 5, alpha=0.1, color='blue', label='Meta Region (||W_V|| < 5)')
    ax.axvspan(15, 30, alpha=0.1, color='red', label='EleutherAI Region (||W_V|| > 15)')

    # Add diagonal reference
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='||W_V|| = ||W_O||')

    ax.set_xlabel('Mean ||$W_V$|| (Value Projection Spectral Norm)')
    ax.set_ylabel('Mean ||$W_O$|| (Output Projection Spectral Norm)')
    ax.set_title('Figure 3: Spectral Signature Correspondence\n10x Difference in ||$W_V$|| Between Labs')

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['EleutherAI'],
               markersize=12, label='EleutherAI (Dampeners)', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['Meta'],
               markersize=12, label='Meta (Expanders)', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['BigScience'],
               markersize=12, label='BigScience (Expanders)', markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['OpenAI'],
               markersize=12, label='OpenAI (Neutral)', markeredgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 80)

    # Add annotation about 10x difference
    ax.annotate('10x difference in ||$W_V$||\n(EleutherAI vs Meta)',
               xy=(15, 5), fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def figure4_layer_dynamics(high_rho_data, save_path):
    """
    Figure 4: Layer-wise Residual Stream Dynamics
    Shows characteristic expansion/dampening patterns
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Select representative models
    dampener = None
    expander = None

    for result in high_rho_data['tested_results']:
        if result['lab'] == 'EleutherAI' and dampener is None:
            dampener = result
        if result['lab'] == 'Meta' and expander is None:
            expander = result
        if dampener and expander:
            break

    # Plot dampener (skip first layer which is embedding)
    ax1 = axes[0]
    gains_d = dampener['all_layer_gains'][1:-1]  # Skip embedding and final LN
    layers_d = list(range(1, len(gains_d) + 1))
    ax1.plot(layers_d, gains_d, 'o-', color=COLORS['EleutherAI'], linewidth=2,
             markersize=6, label=dampener['model'].split('/')[-1])
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax1.fill_between(layers_d, 1.0, gains_d, alpha=0.3, color=COLORS['EleutherAI'])
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Residual Gain')
    ax1.set_title(f"Dampening Profile: {dampener['lab']}\n(Mean Gain: {dampener['gain_no_ln_mean']:.3f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.8, 1.3)

    # Plot expander
    ax2 = axes[1]
    gains_e = expander['all_layer_gains'][1:-1]
    layers_e = list(range(1, len(gains_e) + 1))
    ax2.plot(layers_e, gains_e, 's-', color=COLORS['Meta'], linewidth=2,
             markersize=6, label=expander['model'].split('/')[-1])
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax2.fill_between(layers_e, 1.0, gains_e, alpha=0.3, color=COLORS['Meta'])
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Residual Gain')
    ax2.set_title(f"Expansion Profile: {expander['lab']}\n(Mean Gain: {expander['gain_no_ln_mean']:.3f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.8, 1.3)

    fig.suptitle('Figure 4: Layer-wise Residual Stream Dynamics', fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def figure5_lab_comparison(high_rho_data, save_path):
    """
    Figure 5: Lab Thermodynamic Signatures
    Mean gain with error bars by lab
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    lab_data = high_rho_data['h26_lab_signatures']

    # Sort by mean gain
    labs = sorted(lab_data.keys(), key=lambda x: lab_data[x]['mean_gain'])

    means = [lab_data[lab]['mean_gain'] for lab in labs]
    stds = [lab_data[lab]['std_gain'] for lab in labs]
    signatures = [lab_data[lab]['signature'] for lab in labs]

    x = np.arange(len(labs))
    colors = [COLORS.get(lab, 'gray') for lab in labs]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=1.5)

    # Add horizontal lines
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Neutral (G=1)')
    ax.axhline(y=0.95, color='red', linestyle=':', alpha=0.7)
    ax.axhline(y=1.05, color='blue', linestyle=':', alpha=0.7)

    # Add signature labels
    for i, (sig, mean) in enumerate(zip(signatures, means)):
        ax.annotate(sig, (i, mean + stds[i] + 0.02), ha='center',
                   fontsize=9, fontweight='bold',
                   color='red' if sig == 'DAMPENER' else 'blue' if sig == 'EXPANDER' else 'gray')

    ax.set_xlabel('Research Lab')
    ax.set_ylabel('Mean Residual Gain')
    ax.set_title('Figure 5: Lab Thermodynamic Signatures\nMean Residual Gain by Training Heritage')
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=0)
    ax.legend(loc='upper left')
    ax.set_ylim(0.85, 1.25)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Generate all figures."""
    print("Loading data...")
    pythia_data, high_rho_data, spectral_data = load_results()

    # Output directory
    output_dir = Path(__file__).parent / "Figures"
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating figures...")

    # Figure 1: Kleiber's Law
    figure1_kleiber_law(pythia_data, output_dir / "fig1_kleiber_law.png")

    # Figure 2: Training Heritage
    figure2_training_heritage(high_rho_data, output_dir / "fig2_training_heritage.png")

    # Figure 3: Spectral Signature
    figure3_spectral_signature(spectral_data, output_dir / "fig3_spectral_signature.png")

    # Figure 4: Layer Dynamics
    figure4_layer_dynamics(high_rho_data, output_dir / "fig4_layer_dynamics.png")

    # Figure 5: Lab Comparison
    figure5_lab_comparison(high_rho_data, output_dir / "fig5_lab_comparison.png")

    print(f"\nAll figures saved to: {output_dir}")
    print("\nFigure summary:")
    print("  fig1_kleiber_law.png      - Kleiber's Law scaling (r=-0.878)")
    print("  fig2_training_heritage.png - Lab signatures (EleutherAI: 80% dampen)")
    print("  fig3_spectral_signature.png - W_V/W_O spectral analysis (10x diff)")
    print("  fig4_layer_dynamics.png    - Layer-wise expansion/dampening")
    print("  fig5_lab_comparison.png    - Mean gain by lab")


if __name__ == "__main__":
    main()
