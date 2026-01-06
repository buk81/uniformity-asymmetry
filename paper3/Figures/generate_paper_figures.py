#!/usr/bin/env python3
"""
Paper 3 Figure Generation Script
Thermodynamic Constraints in Transformer Architectures: A Sheaf-Theoretic Perspective

Generates publication-ready figures using Seaborn + adjustText.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from adjustText import adjust_text
import os

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------
# Color palette (given)
# -----------------------
COLORS = {
    'EleutherAI': '#E74C3C',   # Red
    'Meta': '#3498DB',         # Blue
    'BigScience': '#27AE60',   # Green
    'TII': '#F39C12',          # Orange
    'StabilityAI': '#1ABC9C',  # Teal
    'OpenAI': '#8E44AD',       # Purple
}

EXPAND_COLOR = COLORS['Meta']
DAMPEN_COLOR = COLORS['EleutherAI']

# -----------------------
# Data (exact from experiments)
# -----------------------
kleiber_data = {
    'model': ['70m', '160m', '410m', '1b', '1.4b', '2.8b', '6.9b', '12b'],
    'layers': [6, 12, 24, 16, 24, 32, 32, 36],
    'gain': [1.201, 1.157, 0.978, 1.216, 1.005, 0.927, 0.994, 0.986],
    'is_dampening': [False, False, True, False, False, True, True, True]
}
R1, P1 = -0.812, 0.014

heritage_data = {
    'EleutherAI': {'models': ['pythia-1.4b','pythia-2.8b','pythia-6.9b','gpt-neo-1.3b','gpt-j-6b'],
                   'gains': [1.005,0.927,0.994,0.892,1.136], 'mean': 0.991, 'dampen_pct': 80},
    'Meta': {'models': ['opt-125m','opt-350m','opt-1.3b','opt-2.7b','opt-6.7b'],
             'gains': [1.263,0.999,1.090,1.081,1.077], 'mean': 1.102, 'dampen_pct': 20},
    'BigScience': {'models': ['bloom-560m','bloom-1b1','bloom-1b7','bloom-3b'],
                   'gains': [1.276,1.067,1.098,0.998], 'mean': 1.110, 'dampen_pct': 25},
    'TII': {'models': ['falcon-7b'], 'gains': [1.027], 'mean': 1.027, 'dampen_pct': 0},
    'StabilityAI': {'models': ['stablelm-3b'], 'gains': [1.084], 'mean': 1.084, 'dampen_pct': 0},
}

spectral_data = {
    'model': ['pythia-160m', 'pythia-410m', 'gpt-neo-125M', 'opt-125m', 'opt-350m', 'bloom-560m', 'gpt2'],
    'lab': ['EleutherAI', 'EleutherAI', 'EleutherAI', 'Meta', 'Meta', 'BigScience', 'OpenAI'],
    'W_V_mean': [22.15, 16.93, 25.80, 1.61, 2.75, 3.78, 8.95],
    'W_O_mean': [2.39, 2.46, 70.08, 2.02, 3.43, 2.56, 22.51],
    'behavior': ['DAMPEN', 'DAMPEN', 'DAMPEN', 'EXPAND', 'EXPAND', 'EXPAND', 'EXPAND']
}

# -----------------------
# Style (paper clean)
# -----------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})

def clean(ax):
    sns.despine(ax=ax, top=True, right=True)
    ax.grid(True, which="major", alpha=0.20, linewidth=1.0)
    ax.grid(True, which="minor", alpha=0.10, linewidth=0.7)

def add_rp_box(ax, r, p, position="bottom_left"):
    """Add r/p statistics box. Position: 'bottom_left' (safe zone) or 'top_left'."""
    if position == "bottom_left":
        x, y, va = 0.03, 0.08, "bottom"
    else:
        x, y, va = 0.03, 0.96, "top"
    ax.text(
        x, y,
        f"r = {r:.3f}\np = {p:.3f}",
        transform=ax.transAxes,
        ha="left", va=va,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.25", alpha=0.95)
    )

# -----------------------
# Figure 1: Kleiber's Law
# -----------------------
def fig1(ax):
    df = pd.DataFrame(kleiber_data)
    df["regime"] = np.where(df["is_dampening"], "Dampener (G < 1)", "Expander (G > 1)")

    # Points - now using L (depth) on x-axis for better distribution
    for regime, marker, color in [
        ("Expander (G > 1)", '^', EXPAND_COLOR),
        ("Dampener (G < 1)", 'v', DAMPEN_COLOR),
    ]:
        sub = df[df["regime"] == regime]
        ax.scatter(sub["layers"], sub["gain"],
                   s=160, marker=marker, color=color,
                   edgecolors="white", linewidths=1.8, zorder=4)

    # Theoretical bound G_max = 10^(1/L) as function of L
    L_line = np.linspace(5, 40, 300)
    ax.plot(L_line, 10**(1/L_line), "--", color="0.45", lw=2.4, label=r"$G_{\max}=10^{1/L}$")
    ax.axhline(1.0, color="0.25", lw=1.5, alpha=0.5)  # Thinner, lighter neutral line

    # Clean x-limits
    ax.set_xlim(4, 40)
    ax.set_ylim(0.88, 1.25)

    # Manual label offsets - away from markers, no collision
    # Format: model -> (x_offset, y_offset) in points
    label_offsets = {
        '70m':   (8, 6),      # Expander ▲ - right/up
        '160m':  (8, -12),    # Expander ▲ - right/down (avoid curve)
        '1b':    (8, 6),      # Expander ▲ - right/up
        '1.4b':  (8, 8),      # Expander ▲ - right/up
        '410m':  (8, -14),    # Dampener ▼ - right/down
        '2.8b':  (8, -14),    # Dampener ▼ - right/down
        '6.9b':  (8, 8),      # Dampener ▼ - right/up (near 1.0 line)
        '12b':   (8, 8),      # Dampener ▼ - right/up
    }

    for _, row in df.iterrows():
        offset = label_offsets.get(row["model"], (8, 6))
        ax.annotate(
            row["model"],
            (row["layers"], row["gain"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            zorder=5
        )

    add_rp_box(ax, R1, P1)

    ax.set_title("Figure 1: Kleiber's Law for Transformers")
    ax.set_xlabel("Network Depth (L)")
    ax.set_ylabel("Mean Residual Gain (no final LN)")

    legend_elems = [
        Line2D([0],[0], marker='^', color='none', label='Expander (G > 1)',
               markerfacecolor=EXPAND_COLOR, markeredgecolor='white', markersize=9),
        Line2D([0],[0], marker='v', color='none', label='Dampener (G < 1)',
               markerfacecolor=DAMPEN_COLOR, markeredgecolor='white', markersize=9),
        Line2D([0],[0], linestyle='--', color='0.45', label=r"$G_{\max}=10^{1/L}$"),
    ]
    # Legend inside plot (upper right, slightly inward) - safe with L x-axis
    leg = ax.legend(handles=legend_elems, loc="upper right",
                    bbox_to_anchor=(0.98, 0.98), fontsize=10,
                    frameon=True, framealpha=0.95, borderaxespad=0)
    leg.get_frame().set_edgecolor("0.7")

    clean(ax)

# -----------------------
# Figure 2: Heritage Dominance (Summary Figure - NeurIPS-ready)
# -----------------------
def fig2(ax, seed=7):
    """Clean summary figure: Lab means + error bars (n>1 only). n integrated in x-labels."""

    rows = []
    for lab, d in heritage_data.items():
        for m, g in zip(d["models"], d["gains"]):
            rows.append({"lab": lab, "model": m, "gain": g})
    df = pd.DataFrame(rows)

    # Compute stats per lab
    stats = df.groupby("lab")["gain"].agg(["mean", "std", "count"]).reset_index()
    stats["se"] = stats["std"] / np.sqrt(stats["count"])  # Standard error
    stats = stats.sort_values("mean")
    order = stats["lab"].tolist()

    # Subtle zones
    ax.axhspan(0.85, 1.0, color=DAMPEN_COLOR, alpha=0.04, zorder=0)
    ax.axhspan(1.0, 1.30, color=EXPAND_COLOR, alpha=0.03, zorder=0)

    # G=1 reference line (thinner)
    ax.axhline(1.0, color="0.25", lw=1.8, alpha=0.7, zorder=1)

    # Plot means with error bars (colored by lab)
    # NO numerical labels in plot - they go in caption (Option A, NeurIPS-style)
    for i, row in stats.iterrows():
        lab = row["lab"]
        x_pos = order.index(lab)
        color = COLORS.get(lab, "0.4")
        n = int(row["count"])

        # Error bar (±1 SE) - ONLY for n > 1, refined styling
        if n > 1:
            ax.errorbar(x_pos, row["mean"], yerr=row["se"],
                        fmt="none", ecolor="0.35", elinewidth=1.5, capsize=4, capthick=1.5, zorder=3)

        # Diamond marker (colored)
        ax.scatter(x_pos, row["mean"], marker="D", s=200, color=color,
                   edgecolors="white", linewidths=2.0, zorder=4)

    # x-axis labels with n integrated (clean, consistent)
    stats_by_lab = {row["lab"]: int(row["count"]) for _, row in stats.iterrows()}
    xticklabels = [f"{lab}\n(n={stats_by_lab[lab]})" for lab in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xticklabels, fontsize=10)

    ax.set_xlim(-0.6, len(order) - 0.4)
    # Dynamic y-limit: ensure all error bars fully visible
    max_upper = (stats["mean"] + stats["se"]).max()
    min_lower = (stats["mean"] - stats["se"]).min()
    ax.set_ylim(min(0.92, min_lower - 0.02), max(1.18, max_upper + 0.02))
    ax.set_title("Figure 2: Training Heritage Determines Thermodynamic Signature")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Residual Gain (no final LN)")

    clean(ax)

# -----------------------
# Figure 3: Spectral Signature (Cleaned - NeurIPS-ready)
# -----------------------
def fig3(ax):
    """Mechanistic bridge figure: Spectral norms predict thermodynamic behavior."""
    df = pd.DataFrame(spectral_data)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # NO background shading, NO vertical threshold lines (Fix 2, 3)
    # Let the data speak for itself

    # All points = circles, only color = lab (Fix 1: unified marker shape)
    for _, r in df.iterrows():
        ax.scatter(
            r["W_V_mean"], r["W_O_mean"],
            s=180, marker="o",
            color=COLORS.get(r["lab"], "0.3"),
            edgecolors="white", linewidths=1.8,
            zorder=3
        )

    # Manual label offsets - no collisions, paper-stable
    # Format: model -> (x_offset, y_offset) in points
    label_offsets = {
        'pythia-160m':   (10, -16),    # Below (avoid collision with 410m)
        'pythia-410m':   (10, 10),     # Above
        'gpt-neo-125M':  (8, -14),     # Below (outlier, marked with *)
        'opt-125m':      (-60, 6),     # Left (avoid edge)
        'opt-350m':      (10, 8),      # Right/up
        'bloom-560m':    (10, 8),      # Right/up
        'gpt2':          (10, 8),      # Right/up
    }

    for _, r in df.iterrows():
        label = r["model"]
        # Mark GPT-Neo as architectural outlier
        if "gpt-neo" in label.lower():
            label = f"{label}*"

        offset = label_offsets.get(r["model"], (10, 6))
        ax.annotate(
            label,
            (r["W_V_mean"], r["W_O_mean"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
            zorder=4
        )

    # 10x bracket annotation (key finding, positioned between clusters)
    y_arrow = 7
    ax.annotate("", xy=(20, y_arrow), xytext=(2, y_arrow),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=1.6))
    ax.text(5.5, y_arrow * 1.5, r"~10× in $\|W_V\|_2$",
            fontsize=9, color="0.3", fontweight="bold")

    ax.set_xlim(1.0, 100)
    ax.set_ylim(1.5, 100)
    ax.set_title("Figure 3: Spectral Signatures Predict Thermodynamic Behavior")
    ax.set_xlabel(r"Mean $\|W_V\|_2$ (spectral norm)")
    ax.set_ylabel(r"Mean $\|W_O\|_2$ (spectral norm)")

    # Simplified legend: only lab colors (Fix 1: no shape distinction)
    lab_handles = [
        Line2D([0],[0], marker="o", color="none", markerfacecolor=COLORS["EleutherAI"],
               markeredgecolor="white", markersize=9, label="EleutherAI (dampener)"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=COLORS["Meta"],
               markeredgecolor="white", markersize=9, label="Meta (expander)"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=COLORS["BigScience"],
               markeredgecolor="white", markersize=9, label="BigScience (expander)"),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=COLORS["OpenAI"],
               markeredgecolor="white", markersize=9, label="OpenAI (expander)"),
    ]
    ax.legend(handles=lab_handles, loc="upper left", fontsize=9,
              frameon=True, framealpha=0.92)

    # Footnote removed - explanation goes in caption
    # "*GPT-Neo-125M is an architectural outlier with unusually high ||W_O||₂"

    clean(ax)

# -----------------------
# Figure 4 (Appendix): Layer-wise Spectral Dynamics - 2-Panel
# -----------------------
def fig4_2panel():
    """Appendix figure: 2-panel layer-wise W_V spectral norm evolution.

    Panel A: All models (shows EleutherAI spike)
    Panel B: Non-EleutherAI only (shows flat/low regime in detail)
    """
    import json

    # Load layer-wise data
    results_path = os.path.join(os.path.dirname(OUTPUT_DIR), "Results",
                                "restriction_map_spectral_20260105_202448.json")
    with open(results_path) as f:
        data = json.load(f)

    # Models by category
    eleuther_models = [
        ("EleutherAI/pythia-160m", "pythia-160m"),
        ("EleutherAI/pythia-410m", "pythia-410m"),
    ]
    other_models = [
        ("facebook/opt-125m", "opt-125m"),
        ("facebook/opt-350m", "opt-350m"),
        ("bigscience/bloom-560m", "bloom-560m"),
        ("gpt2", "gpt2"),
    ]

    # Line styles by lab (base styles)
    lab_styles = {
        'Meta': {'ls': '--', 'lw': 2.0, 'alpha': 0.85},
        'BigScience': {'ls': '-.', 'lw': 2.0, 'alpha': 0.85},
        'OpenAI': {'ls': ':', 'lw': 2.4, 'alpha': 0.9},
    }

    # DISTINCT styles for EleutherAI models (different line styles + shades)
    eleuther_styles = {
        'pythia-160m': {'ls': '-', 'lw': 2.4, 'alpha': 0.95, 'color': '#E74C3C'},   # Solid, bright red
        'pythia-410m': {'ls': '--', 'lw': 2.2, 'alpha': 0.9, 'color': '#B03A2E'},   # Dashed, dark red
    }

    # Create 2-panel figure
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.2))

    # Storage for direct labels
    eleuther_endpoints = {}

    # Helper to plot a model
    def plot_model(ax, model_full, label_override=None, store_endpoint=False):
        entry = next((e for e in data["results"] if e["model"] == model_full), None)
        if entry is None:
            return None, None
        model_name = label_override or model_full.split("/")[-1]
        lab = entry["lab"]
        w_v = entry["W_V_spectral_norm"]
        layers_norm = [(l+1) / len(w_v) for l in range(len(w_v))]

        # Use model-specific style for EleutherAI, otherwise lab style
        if model_name in eleuther_styles:
            style = eleuther_styles[model_name]
            color = style.pop('color', COLORS.get(lab, "0.5"))
            ax.plot(layers_norm, w_v, color=color, label=model_name, **style)
            style['color'] = color  # restore for reuse
        else:
            style = lab_styles.get(lab, {'ls': '-', 'lw': 1.5, 'alpha': 0.7})
            ax.plot(layers_norm, w_v, color=COLORS.get(lab, "0.5"),
                    label=model_name, **style)

        if store_endpoint:
            return layers_norm[-1], w_v[-1]
        return None, None

    # ===== PANEL A: All models =====
    for model_full, label in eleuther_models:
        x_end, y_end = plot_model(ax_a, model_full, label, store_endpoint=True)
        if x_end is not None:
            eleuther_endpoints[label] = (x_end, y_end)

    for model_full, label in other_models:
        plot_model(ax_a, model_full, label)

    ax_a.axhline(10, color="0.4", lw=1.2, alpha=0.4, ls=":")
    ax_a.set_xlim(0, 1.18)  # Extra space for direct labels
    ax_a.set_ylim(0, 70)
    ax_a.set_xlabel("Relative Layer Position (l/L)")
    ax_a.set_ylabel(r"$\|W_V\|_2$ (spectral norm)")

    # DIRECT LABELS for EleutherAI models (right edge, no legend confusion)
    for model_name, (x_end, y_end) in eleuther_endpoints.items():
        color = eleuther_styles[model_name]['color']
        # Offset to avoid overlap
        y_offset = 3 if '160m' in model_name else -5
        ax_a.text(x_end + 0.02, y_end + y_offset, model_name,
                  fontsize=9, color=color, va='center', fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # Remove EleutherAI from legend (they have direct labels now)
    handles, labels = ax_a.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if 'pythia' not in l]
    ax_a.legend([h for h, l in filtered], [l for h, l in filtered],
                loc="upper left", fontsize=8, frameon=True, framealpha=0.92,
                ncol=2, columnspacing=0.8, title="Non-EleutherAI", title_fontsize=8)

    clean(ax_a)

    # ===== PANEL B: Non-EleutherAI only =====
    # DISTINCT styles for same-lab models
    model_styles = {
        'opt-125m': {'ls': '-', 'lw': 2.2, 'color': '#3498DB'},    # Solid, bright blue
        'opt-350m': {'ls': '--', 'lw': 2.0, 'color': '#1A5276'},   # Dashed, dark blue
        'bloom-560m': {'ls': '-.', 'lw': 2.0, 'color': COLORS['BigScience']},
        'gpt2': {'ls': ':', 'lw': 2.4, 'color': COLORS['OpenAI']},
    }

    panel_b_endpoints = {}
    for model_full, label in other_models:
        entry = next((e for e in data["results"] if e["model"] == model_full), None)
        if entry is None:
            continue
        w_v = entry["W_V_spectral_norm"]
        layers_norm = [(l+1) / len(w_v) for l in range(len(w_v))]

        style = model_styles.get(label, {'ls': '-', 'lw': 1.5, 'color': '0.5'})
        ax_b.plot(layers_norm, w_v, label=label, **style)
        panel_b_endpoints[label] = (layers_norm[-1], w_v[-1], style['color'])

    ax_b.axhline(10, color="0.4", lw=1.2, alpha=0.4, ls=":")
    ax_b.set_xlim(0, 1.18)  # Extra space for direct labels
    ax_b.set_ylim(0, 12)
    ax_b.set_xlabel("Relative Layer Position (l/L)")
    ax_b.set_ylabel(r"$\|W_V\|_2$ (spectral norm)")

    # DIRECT LABELS for all models (right edge)
    label_offsets = {
        'opt-125m': -0.3,      # Below
        'opt-350m': 0.5,       # Above
        'bloom-560m': -0.8,    # Below (ends around 5)
        'gpt2': 0.5,           # Above (ends around 11)
    }
    for model_name, (x_end, y_end, color) in panel_b_endpoints.items():
        y_off = label_offsets.get(model_name, 0)
        ax_b.text(x_end + 0.02, y_end + y_off, model_name,
                  fontsize=9, color=color, va='center', fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # No legend needed - all models have direct labels
    clean(ax_b)

    # Panel labels (A/B)
    ax_a.text(-0.08, 1.05, "A", transform=ax_a.transAxes,
              fontsize=14, fontweight="bold", va="bottom")
    ax_b.text(-0.08, 1.05, "B", transform=ax_b.transAxes,
              fontsize=14, fontweight="bold", va="bottom")

    fig.suptitle("Appendix: Layer-wise Spectral Dynamics", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


# Legacy single-axis version (not used)
def fig4(ax):
    """Legacy: single-axis version. Use fig4_2panel() instead."""
    pass


# -----------------------
# Figure A2 (Appendix): L* Predicted vs Empirical
# -----------------------
def fig_a2_lstar():
    """Appendix figure: L* formula validation across architectures.

    Shows predicted vs empirical L* values with error bars.
    Formula: L* = L × (0.11 + 0.012×L + 4.9/H)
    """
    import json

    # Load L* validation data
    results_path = os.path.join(os.path.dirname(OUTPUT_DIR), "Results",
                                "l_star_cross_heritage_20260106_104357.json")
    with open(results_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Extract data
    models = []
    for r in data["results"]:
        models.append({
            'name': r["model"].split("/")[-1],
            'lab': r["lab"],
            'L': r["L"],
            'H': r["H"],
            'predicted': r["L_star_predicted"],
            'empirical': r["L_star_empirical"],
            'std': r["L_star_std"],
            'error_pct': r["error_pct"]
        })

    # Perfect prediction line (y=x)
    ax.plot([0, 20], [0, 20], '--', color='0.4', lw=1.8, alpha=0.8, zorder=1)

    # Plot each model - error bars subtle (empirical uncertainty, not prediction)
    for m in models:
        color = COLORS.get(m['lab'], '0.5')
        # Error bar first (behind point) - subtle gray, not colored
        if m['std'] > 0:
            ax.errorbar(m['predicted'], m['empirical'], yerr=m['std'],
                        fmt='none', ecolor='0.55', elinewidth=1.2, capsize=3, capthick=1.0,
                        alpha=0.7, zorder=2)
        # Point on top
        ax.scatter(m['predicted'], m['empirical'], s=120, color=color,
                   edgecolor='white', linewidth=1.8, zorder=3)

    # Direct labels - RULE: labels must not be between point and diagonal
    # Points BELOW diagonal (empirical < predicted): label goes DOWN/LEFT
    # Points ABOVE diagonal (empirical > predicted): label goes UP/RIGHT
    label_offsets = {
        'pythia-160m': (0.4, -1.2),      # Below diag → label DOWN
        'pythia-410m': (0.4, -1.0),      # Below diag → label DOWN (not between point & line!)
        'opt-125m': (-2.8, 0.5),         # Above diag → label LEFT/UP
        'opt-350m': (0.4, 1.0),          # Near diag → label UP
        'bloom-560m': (0.4, -1.5),       # Far below → label DOWN
        'gpt2': (-2.0, 0.8),             # Above diag → label LEFT/UP
    }
    for m in models:
        color = COLORS.get(m['lab'], '0.5')
        off = label_offsets.get(m['name'], (0.3, 0.3))
        ax.annotate(m['name'], (m['predicted'] + off[0], m['empirical'] + off[1]),
                    fontsize=9, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    # Formatting
    ax.set_xlim(5, 19)
    ax.set_ylim(3, 18)
    ax.set_xlabel(r"$L^*$ Predicted")
    ax.set_ylabel(r"$L^*$ Empirical")
    ax.set_title(f"Appendix: L* Formula Validation (MAPE = {data['overall_mape']:.1f}%)")

    # Stats annotation
    ax.text(0.05, 0.95, f"Formula: $L^* = L \\times (0.11 + 0.012L + 4.9/H)$\nn = {data['n_models']} models",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.6', alpha=0.92))

    # Legend for labs
    lab_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['EleutherAI'],
               markeredgecolor='white', markersize=9, label='EleutherAI'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['Meta'],
               markeredgecolor='white', markersize=9, label='Meta'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['BigScience'],
               markeredgecolor='white', markersize=9, label='BigScience'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['OpenAI'],
               markeredgecolor='white', markersize=9, label='OpenAI'),
    ]
    ax.legend(handles=lab_handles, loc='lower right', fontsize=9, frameon=True, framealpha=0.92)

    clean(ax)
    fig.tight_layout()
    return fig


# -----------------------
# Figure A3 (Appendix): Pythia Family Scaling
# -----------------------
def fig_a3_pythia():
    """Appendix figure: Full Pythia family (8 models) showing Kleiber's Law.

    Demonstrates that deeper models are constrained to operate near G=1 or below.
    """
    import json

    # Load Pythia family data
    results_path = os.path.join(os.path.dirname(OUTPUT_DIR), "Results",
                                "pythia_family_NO_FINAL_LN_20260105_174820.json")
    with open(results_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Extract model data
    models = []
    for m in data["models"]:
        models.append({
            'name': m["model"],
            'L': m["n_layers"],
            'G': m["residual_gain_mean"],
            'G_std': m["residual_gain_std"],
            'rho': m["rho"],
            'is_dampening': m["is_dampening"]
        })

    # Sort by depth for connecting line
    models.sort(key=lambda x: x['L'])

    # Kleiber bound: G_max = 10^(1/L)
    L_range = np.linspace(5, 40, 100)
    G_max = 10 ** (1 / L_range)
    ax.plot(L_range, G_max, '--', color='0.5', lw=1.8, alpha=0.7,
            label=r'$G_{max} = 10^{1/L}$')

    # Neutrality line
    ax.axhline(1.0, color='0.4', lw=1.2, ls=':', alpha=0.6)

    # Plot each model
    for m in models:
        color = COLORS['EleutherAI'] if not m['is_dampening'] else '#8B0000'  # Dark red for dampeners
        marker = '^' if not m['is_dampening'] else 'v'  # Up for expander, down for dampener

        # Error bar - LIGHTER for 70m (largest uncertainty), subtle for others
        eb_alpha = 0.35 if '70m' in m['name'] else 0.55
        eb_lw = 1.0 if '70m' in m['name'] else 1.2
        ax.errorbar(m['L'], m['G'], yerr=m['G_std'],
                    fmt='none', ecolor='0.55', elinewidth=eb_lw, capsize=3, alpha=eb_alpha, zorder=2)

        # Point
        ax.scatter(m['L'], m['G'], s=140, c=color, marker=marker,
                   edgecolor='white', linewidth=1.5, zorder=3)

    # Direct labels - positioned CLOSE TO MARKER (not at curve!)
    # Rule: label distance < errorbar length, labels reference POINTS not theory
    label_positions = {
        'pythia-70m': (0.8, 0.015),      # Right, slightly above marker (NOT at curve!)
        'pythia-160m': (1.2, 0.015),     # Right, above errorbar
        'pythia-410m': (1.2, -0.025),    # Right, below (dampener)
        'pythia-1b': (1.2, 0.035),       # Right, above errorbar
        'pythia-1.4b': (1.2, 0.02),      # Right, above
        'pythia-2.8b': (1.2, -0.035),    # Right, below (dampener)
        'pythia-6.9b': (1.2, 0.02),      # Right, above
        'pythia-12b': (1.2, 0.02),       # Right, above
    }
    for m in models:
        off = label_positions.get(m['name'], (1.2, 0.02))
        short_name = m['name'].replace('pythia-', '')
        color = COLORS['EleutherAI'] if not m['is_dampening'] else '#8B0000'
        ax.annotate(short_name, (m['L'] + off[0], m['G'] + off[1]),
                    fontsize=9, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85))

    # Formatting - ylim=1.42 ensures full 70m errorbar (G=1.201, std=0.176 → upper=1.377)
    ax.set_xlim(4, 40)
    ax.set_ylim(0.88, 1.42)
    ax.set_xlabel("Network Depth (L)")
    ax.set_ylabel("Mean Residual Gain (G)")
    ax.set_title("Appendix: Pythia Family Scaling (Kleiber's Law)")

    # Annotations for regions - centered, in safe zones (ylim 0.88-1.42)
    ax.text(28, 1.38, "EXPANSION (G > 1)", fontsize=10, ha='center', color='0.4', style='italic')
    ax.text(28, 0.90, "DAMPENING (G < 1)", fontsize=10, ha='center', color='0.4', style='italic')

    # Stats box - bottom left (safe zone)
    corr = data["analysis"]["rho_gain_correlation"]
    p_val = data["analysis"]["correlation_p_value"]
    ax.text(0.03, 0.03, f"Depth–Gain: r = {corr:.2f}, p = {p_val:.3f}",
            transform=ax.transAxes, fontsize=9, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.6', alpha=0.92))

    # Legend - MINIMAL (plot text shows regimes, legend just for markers/curve)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='none', markerfacecolor=COLORS['EleutherAI'],
               markeredgecolor='white', markersize=9, label='Expander'),
        Line2D([0], [0], marker='v', color='none', markerfacecolor='#8B0000',
               markeredgecolor='white', markersize=9, label='Dampener'),
        Line2D([0], [0], ls='--', color='0.5', lw=1.5, label=r'$G_{max}$'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True, framealpha=0.92)

    clean(ax)
    fig.tight_layout()
    return fig


# -----------------------
# Figure A4 (Appendix): Input Robustness
# -----------------------
def fig_a4_input_robustness():
    """Appendix figure: Thermodynamic signature robustness across inputs.

    Shows gain distributions (25 prompts) for all Pythia models.
    Key message: Dampeners remain dampeners despite input variation.
    Outliers beyond ylim are clipped and shown at boundary.
    """
    import json

    # Load Pythia family data with all 25 prompt gains
    results_path = os.path.join(os.path.dirname(OUTPUT_DIR), "Results",
                                "pythia_family_NO_FINAL_LN_20260105_174820.json")
    with open(results_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(11, 6))

    # Y-axis limits for clean display (outliers will be clipped)
    # 0.82 gives comfortable headroom for bottom outlier marker
    Y_MIN, Y_MAX = 0.82, 1.38

    # Sort models by depth
    models_sorted = sorted(data["models"], key=lambda x: x["n_layers"])
    model_order = [m["model"].replace("pythia-", "") for m in models_sorted]

    # Color palette: red for expanders, dark gray for dampeners (better contrast)
    palette = {}
    palette_diamond = {}  # Separate palette for diamonds (dampeners = gray)
    for m in models_sorted:
        name = m["model"].replace("pythia-", "")
        if m["is_dampening"]:
            palette[name] = '#8B0000'      # Points: dark red
            palette_diamond[name] = '#2C2C2C'  # Diamond: dark gray (semantic: dampening)
        else:
            palette[name] = COLORS['EleutherAI']  # Points: red
            palette_diamond[name] = COLORS['EleutherAI']  # Diamond: red (semantic: expansion)

    # Neutrality line FIRST (behind everything)
    ax.axhline(1.0, color='0.4', lw=1.5, ls=':', alpha=0.6, zorder=1)

    # Plot each model manually (for outlier handling)
    np.random.seed(42)
    outlier_count = 0

    for i, m in enumerate(models_sorted):
        name = m["model"].replace("pythia-", "")
        gains = np.array(m["residual_gain_all"])
        color = palette[name]

        # Jitter x positions
        x_jitter = np.random.uniform(-0.25, 0.25, len(gains))

        # Separate in-range and outliers
        in_range = (gains >= Y_MIN) & (gains <= Y_MAX)
        below = gains < Y_MIN
        above = gains > Y_MAX

        # Plot in-range points (very transparent)
        ax.scatter(i + x_jitter[in_range], gains[in_range],
                   s=30, color=color, alpha=0.25, zorder=2, edgecolors='none')

        # Plot outliers at boundary with different marker (triangles)
        if below.any():
            ax.scatter(i + x_jitter[below], [Y_MIN + 0.01] * below.sum(),
                       s=40, color=color, alpha=0.6, marker='v', zorder=2, edgecolors='none')
            outlier_count += below.sum()
        if above.any():
            ax.scatter(i + x_jitter[above], [Y_MAX - 0.01] * above.sum(),
                       s=40, color=color, alpha=0.6, marker='^', zorder=2, edgecolors='none')
            outlier_count += above.sum()

        # Mean + SD overlay (bold)
        mean_g = gains.mean()
        std_g = gains.std()

        # Clip mean display if outside range
        mean_display = np.clip(mean_g, Y_MIN, Y_MAX)

        # Error bar (clipped to ylim)
        err_low = min(std_g, mean_g - Y_MIN) if mean_g - std_g < Y_MIN else std_g
        err_high = min(std_g, Y_MAX - mean_g) if mean_g + std_g > Y_MAX else std_g

        # Error bar (subtle, not dominant)
        ax.errorbar(i, mean_display, yerr=[[err_low], [err_high]], fmt='none',
                    ecolor='0.45', elinewidth=1.5, capsize=3, capthick=1.2, alpha=0.6, zorder=3)

        # Diamond marker (bold, high contrast) - use semantic color
        diamond_color = palette_diamond[name]
        ax.scatter(i, mean_display, marker='D', s=120, color=diamond_color,
                   edgecolor='white', linewidth=2.0, zorder=4)

    # X-axis labels (compact)
    depth_labels = [f"{m['model'].replace('pythia-', '')}\n({m['n_layers']})" for m in models_sorted]
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(depth_labels, fontsize=9)

    # Formatting
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(-0.6, len(model_order) - 0.4)
    ax.set_xlabel("Model (Depth L)")
    ax.set_ylabel("Residual Gain (G)")
    ax.set_title("Appendix: Input Robustness of Thermodynamic Signature")

    # Regime annotations (subtle, not competing with data)
    ax.text(0.5, 1.32, "EXPANSION (G > 1)", fontsize=9, ha='left', color='0.55', style='italic', alpha=0.8)
    ax.text(5.5, 0.84, "DAMPENING (G < 1)", fontsize=9, ha='left', color='0.55', style='italic', alpha=0.8)

    # Legend - clear symbol explanation (semantic colors for expander/dampener)
    n_prompts = len(data["models"][0]["residual_gain_all"])
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='0.6',
               markeredgecolor='none', markersize=7, alpha=0.3, label=f'Individual prompts (n={n_prompts})'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor=COLORS['EleutherAI'],
               markeredgecolor='white', markersize=9, label='Mean (expander)'),
        Line2D([0], [0], marker='D', color='none', markerfacecolor='#2C2C2C',
               markeredgecolor='white', markersize=9, label='Mean (dampener)'),
        Line2D([0], [0], color='0.35', lw=1.8, label='Error bars: ±1 SD'),
    ]
    if outlier_count > 0:
        legend_elements.append(
            Line2D([0], [0], marker='^', color='none', markerfacecolor='0.5',
                   markeredgecolor='none', markersize=7, alpha=0.6, label=f'Outliers clipped ({outlier_count})'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True, framealpha=0.95)

    clean(ax)
    fig.tight_layout()
    return fig


# -----------------------
# Figure A5 (Appendix): Sheaf Laplacian Trace
# -----------------------
def fig_a5_sheaf_trace():
    """Appendix figure: Sheaf Laplacian trace validation.

    Shows layer-wise Tr(L_F) proxy (||W_V||²) across models.
    Key finding: EleutherAI has ~300× higher trace in late layers.
    Uses LOG SCALE to show differences fairly across magnitudes.
    """
    import json

    # Load spectral data
    results_path = os.path.join(os.path.dirname(OUTPUT_DIR), "Results",
                                "restriction_map_spectral_20260105_202448.json")
    with open(results_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Models to show (representative from each lab) - correct names from data
    # OPT-125m lighter (secondary), OPT-350m main representative
    models_to_plot = [
        ("EleutherAI/pythia-160m", "pythia-160m", COLORS['EleutherAI'], '-', 2.4, 0.9),
        ("EleutherAI/pythia-410m", "pythia-410m", '#B03A2E', '--', 2.0, 0.85),
        ("facebook/opt-125m", "opt-125m", '#7FB3D5', '-', 1.5, 0.5),  # Lighter, thinner
        ("facebook/opt-350m", "opt-350m", COLORS['Meta'], '-', 2.2, 0.85),  # Main OPT
        ("bigscience/bloom-560m", "bloom-560m", COLORS['BigScience'], '-', 2.0, 0.85),
        ("gpt2", "gpt2", COLORS['OpenAI'], '-', 2.2, 0.85),
    ]

    # Store endpoints for annotations
    endpoints = {}

    for model_full, label, color, ls, lw, alpha in models_to_plot:
        # Find model in results
        entry = next((r for r in data["results"] if r["model"] == model_full), None)
        if entry is None:
            continue

        # Compute Tr(L_F) proxy: ||W_V||² (squared spectral norm)
        w_v = np.array(entry["W_V_spectral_norm"])
        trace_proxy = w_v ** 2  # ||W_V||² as trace proxy

        # Normalize layer position
        layers_norm = np.linspace(0, 1, len(trace_proxy))

        ax.plot(layers_norm, trace_proxy, color=color, ls=ls, lw=lw, alpha=alpha)

        # Store endpoint for direct label
        endpoints[label] = {
            'x_end': layers_norm[-1],
            'y_end': trace_proxy[-1],
            'color': color
        }

    # LOG SCALE - shows differences fairly across magnitudes
    ax.set_yscale('log')

    # Direct labels at endpoints (log-scale appropriate positions)
    # Second value is multiplicative offset for log scale
    label_offsets = {
        'pythia-160m': (0.02, 1.15),
        'pythia-410m': (0.02, 1.25),
        'opt-125m': (0.02, 0.45),        # BELOW endpoint (avoid opt-350m collision)
        'opt-350m': (0.02, 1.6),         # ABOVE endpoint
        'bloom-560m': (0.02, 1.5),
        'gpt2': (0.02, 1.3),
    }

    for label, ep in endpoints.items():
        x_off, y_mult = label_offsets.get(label, (0.02, 1.0))
        ax.text(ep['x_end'] + x_off, ep['y_end'] * y_mult, label,
                fontsize=9, color=ep['color'], fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))

    # Formatting
    ax.set_xlim(-0.02, 1.15)
    ax.set_ylim(1, 6000)
    ax.set_xlabel("Relative Layer Position (l/L)")
    ax.set_ylabel(r"Trace Proxy: $\|W_V\|_2^2$ (log scale)")
    ax.set_title("Appendix: Sheaf Laplacian Trace by Training Heritage")

    # Single annotation (subtle, dezent) - let caption do the heavy lifting
    ax.text(0.55, 4200, "late-layer spike",
            fontsize=8, ha='left', color='0.45', style='italic', alpha=0.6)

    # Legend (compact) - lab grouping
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['EleutherAI'], lw=2.2, label='EleutherAI (Pythia)'),
        Line2D([0], [0], color=COLORS['Meta'], lw=2.2, label='Meta (OPT)'),
        Line2D([0], [0], color=COLORS['BigScience'], lw=2.0, label='BigScience (BLOOM)'),
        Line2D([0], [0], color=COLORS['OpenAI'], lw=2.2, label='OpenAI (GPT-2)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, frameon=True, framealpha=0.95)

    clean(ax)
    fig.tight_layout()
    return fig


# -----------------------
# Save individual + combined
# -----------------------
def save_all():
    print(f"Saving figures to: {OUTPUT_DIR}")

    # Figure 1
    f, ax = plt.subplots(figsize=(7.8, 5.4))
    fig1(ax)
    f.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_kleiber_law.png")
    f.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(f)
    print(f"  Saved: {path}")

    # Figure 2
    f, ax = plt.subplots(figsize=(7.8, 5.4))
    fig2(ax)
    f.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_training_heritage.png")
    f.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(f)
    print(f"  Saved: {path}")

    # Figure 3
    f, ax = plt.subplots(figsize=(7.8, 5.4))
    fig3(ax)
    f.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_spectral_signature.png")
    f.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(f)
    print(f"  Saved: {path}")

    # Figure A1 (Appendix) - Layer-wise Spectral Dynamics
    fig = fig4_2panel()
    path = os.path.join(OUTPUT_DIR, "fig_a1_layer_dynamics.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Figure A2 (Appendix) - L* Validation
    fig = fig_a2_lstar()
    path = os.path.join(OUTPUT_DIR, "fig_a2_lstar_validation.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Figure A3 (Appendix) - Pythia Family Scaling
    fig = fig_a3_pythia()
    path = os.path.join(OUTPUT_DIR, "fig_a3_pythia_scaling.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Figure A4 (Appendix) - Input Robustness
    fig = fig_a4_input_robustness()
    path = os.path.join(OUTPUT_DIR, "fig_a4_input_robustness.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Figure A5 (Appendix) - Sheaf Laplacian Trace
    fig = fig_a5_sheaf_trace()
    path = os.path.join(OUTPUT_DIR, "fig_a5_sheaf_trace.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")

    # Combined (NeurIPS/ICLR style: no subplot titles, only A/B/C labels)
    fig, axes = plt.subplots(1, 3, figsize=(18.8, 5.2))
    fig1(axes[0])
    fig2(axes[1])
    fig3(axes[2])

    # Remove subplot titles (they go in the caption)
    for ax in axes:
        ax.set_title("")

    # Add panel labels (A/B/C) - positioned above y-axis, no collision
    for ax, letter in zip(axes, ["A", "B", "C"]):
        ax.text(-0.02, 1.12, letter, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="left")

    # Single figure title
    fig.suptitle("Thermodynamic Constraints in Transformer Architectures",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUTPUT_DIR, "fig_combined.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")

    print("\nAll figures generated successfully!")

if __name__ == "__main__":
    save_all()
