#!/usr/bin/env python3
"""
Generate neutral version of Scientific Facts: UA vs Output Preference plot.
Version 3: Fixed spacing, no overlapping labels.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from correlation_analysis.json -> rlhf_masking (Scientific Facts category)
data = {
    'Llama': {'ua': -0.0125, 'pref': 1.165, 'color': '#1f77b4'},
    'Gemma': {'ua': 0.021, 'pref': 1.138, 'color': '#ff7f0e'},
    'Mistral': {'ua': 0.010, 'pref': 0.972, 'color': '#2ca02c'},
    'Apertus': {'ua': 0.109, 'pref': 1.278, 'color': '#d62728'},
}

# Create figure with more vertical space
fig, ax = plt.subplots(figsize=(9, 7))

# Plot each model with direct labels
for model, vals in data.items():
    ua_scaled = vals['ua'] * 10
    ax.scatter(ua_scaled, vals['pref'], s=200, c=vals['color'],
               edgecolors='black', linewidths=1.5, zorder=5)

    # Direct label at each point
    offset_x = 0.08 if model != 'Llama' else -0.08
    ha = 'left' if model != 'Llama' else 'right'
    ax.text(ua_scaled + offset_x, vals['pref'], model,
            fontsize=10, fontweight='bold', ha=ha, va='center')

# Reference lines only (no shading)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Callout annotations - repositioned to avoid overlap
ax.annotate(
    'Low UA\nHigh Output Pref',
    xy=(-0.125, 1.165),
    xytext=(-0.35, 1.32),
    fontsize=9,
    ha='center',
    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='gray', alpha=0.95)
)

ax.annotate(
    'High UA\nHigh Output Pref',
    xy=(1.09, 1.278),
    xytext=(0.75, 1.38),
    fontsize=9,
    ha='center',
    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='gray', alpha=0.95)
)

# Labels
ax.set_xlabel('Uniformity Asymmetry (UA × 10)', fontsize=11, fontweight='bold')
ax.set_ylabel('Output Preference (mean ΔNLL)', fontsize=11, fontweight='bold')

# Title with proper spacing
fig.suptitle('Scientific Facts: Uniformity Asymmetry vs Output Preference',
             fontsize=14, fontweight='bold', y=0.96)

# Subtitle with proper spacing below title
ax.set_title('Llama shows near-zero UA despite strong output preference',
             fontsize=10, style='italic', color='#555555', pad=10)

# Grid
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)

# Axis limits - extended to prevent overlap
ax.set_xlim(-0.55, 1.5)
ax.set_ylim(0.88, 1.45)

# Add note about baseline - repositioned to bottom right
ax.text(0.98, 0.02, 'Dashed lines: UA=0 (neutral), Output=1.0 (no preference)',
        transform=ax.transAxes, fontsize=8, color='gray', style='italic', ha='right')

# Adjust layout
plt.subplots_adjust(top=0.88)

# Save
output_path = 'scientific_facts_ua_vs_output.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")

plt.close()
