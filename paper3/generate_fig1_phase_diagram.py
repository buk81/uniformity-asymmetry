#!/usr/bin/env python3
"""
Figure 1: Thermodynamic Phase Diagram
Based on Gemini's concept, using OUR validated data.

Shows models grouped by lab lineage with gain.
Pure matplotlib implementation (no seaborn dependency).
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# Load our ACTUAL validated data
results_dir = Path(__file__).parent / "Results"

with open(results_dir / "high_rho_hunt_NO_FINAL_LN_20260105_184728.json") as f:
    high_rho = json.load(f)

# Color palette for labs
LAB_COLORS = {
    'EleutherAI': '#E74C3C',    # Red (Dampeners)
    'Meta': '#3498DB',           # Blue
    'BigScience': '#27AE60',     # Green
    'OpenAI': '#9B59B6',         # Purple
    'TII': '#F39C12',            # Orange
    'StabilityAI': '#1ABC9C',    # Teal
}

# Organize data by lab
lab_data = defaultdict(list)
for result in high_rho['tested_results']:
    model_name = result['model'].split('/')[-1]
    lab = result['lab']
    gain = result['gain_no_ln_mean']
    lab_data[lab].append({
        'model': model_name,
        'gain': gain,
        'is_dampening': gain < 1.0
    })

# Sort labs by mean gain
lab_means = {lab: np.mean([m['gain'] for m in models]) for lab, models in lab_data.items()}
lab_order = sorted(lab_means.keys(), key=lambda x: lab_means[x])

print("Data by Lab (sorted by mean gain):")
for lab in lab_order:
    models = lab_data[lab]
    gains = [m['gain'] for m in models]
    print(f"  {lab}: mean={np.mean(gains):.3f}, std={np.std(gains):.3f}, n={len(gains)}")

# ============ PLOTTING ============
fig, ax = plt.subplots(figsize=(12, 7))

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
})

# Phase Zone Shading (draw first, behind everything)
ax.axhspan(0.85, 0.98, alpha=0.15, color='blue', zorder=0, label='_')
ax.axhspan(0.98, 1.02, alpha=0.15, color='gray', zorder=0, label='_')
ax.axhspan(1.02, 1.20, alpha=0.15, color='red', zorder=0, label='_')

# Plot each lab's models as scatter with jitter
for i, lab in enumerate(lab_order):
    models = lab_data[lab]
    n = len(models)

    # Add jitter for visibility
    jitter = np.random.uniform(-0.15, 0.15, n)
    x_positions = i + jitter
    y_positions = [m['gain'] for m in models]
    colors = [LAB_COLORS.get(lab, 'gray')] * n

    # Plot points
    ax.scatter(x_positions, y_positions,
              c=colors, s=150, alpha=0.8,
              edgecolor='black', linewidth=1.5, zorder=5,
              marker='v' if lab_means[lab] < 1.0 else '^')

    # Add model name labels
    for j, m in enumerate(models):
        short_name = m['model'].replace('pythia-', '').replace('opt-', 'O').replace('bloom-', 'B').replace('falcon-', 'F').replace('stablelm-', 'S')
        # Alternate left/right to avoid overlap
        x_off = 0.2 if j % 2 == 0 else -0.2
        ha = 'left' if j % 2 == 0 else 'right'
        ax.annotate(short_name, (x_positions[j], y_positions[j]),
                   xytext=(x_off, 0), textcoords='offset fontsize',
                   fontsize=8, alpha=0.7, ha=ha, va='center')

# Bentov Limit / Neutral Line
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=3, alpha=0.8, zorder=3)
ax.text(len(lab_order) - 0.3, 1.005, 'G = 1.0 (Neutral)',
        fontsize=11, color='black', va='bottom', ha='right', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

# Phase Labels (left side)
ax.text(-0.6, 0.91, 'DAMPENING\n(G < 1)', fontsize=11, color='#1a5276',
        fontweight='bold', va='center', ha='left',
        bbox=dict(boxstyle='round', facecolor='#d4e6f1', alpha=0.8))
ax.text(-0.6, 1.10, 'EXPANSION\n(G > 1)', fontsize=11, color='#922b21',
        fontweight='bold', va='center', ha='left',
        bbox=dict(boxstyle='round', facecolor='#fadbd8', alpha=0.8))

# Lab signature annotations (below x-axis)
for i, lab in enumerate(lab_order):
    sig = high_rho['h26_lab_signatures'].get(lab, {})
    dampen_pct = sig.get('dampen_pct', 0)
    signature = sig.get('signature', 'UNKNOWN')

    color = '#1a5276' if signature == 'DAMPENER' else '#922b21' if signature == 'EXPANDER' else 'gray'
    ax.text(i, 0.86, f'{signature}',
            ha='center', va='top', fontsize=9, color=color, fontweight='bold')

# X-axis: Lab names
ax.set_xticks(range(len(lab_order)))
ax.set_xticklabels(lab_order, fontsize=11, fontweight='bold')

# Titles and Labels
ax.set_title('Figure 1: Thermodynamic Phase Diagram\n'
             'Training Heritage Determines Dynamical Signature (H26)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Mean Residual Gain $G$ (No Final LN)', fontsize=12)
ax.set_xlabel('Research Lab (Training Heritage)', fontsize=12)

# Y-axis limits
ax.set_ylim(0.85, 1.18)
ax.set_xlim(-0.8, len(lab_order) - 0.3)

# Grid
ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
ax.xaxis.grid(False)

# Legend for markers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='v', color='w', markerfacecolor='#E74C3C',
           markersize=12, label='Dampener (mean G < 1)', markeredgecolor='black'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='#3498DB',
           markersize=12, label='Expander (mean G > 1)', markeredgecolor='black'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()

# Save
output_path = Path(__file__).parent / "Figures" / "fig1_phase_diagram.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")
