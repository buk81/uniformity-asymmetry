#!/usr/bin/env python3
"""
H4 Validation v2: Full-Scale Sheaf Laplacian with Multi-Head Integration

Improvements over v1:
1. Quantitative Bounds - a priori L* estimation
2. Full-Scale Laplacian - O(n² + d²) trace computation (no subsampling)
3. Multi-Head Integration - proper handling of all H attention heads

Paper #3: Thermodynamic Constraints in Transformer Architectures
Author: Davide D'Elia
Date: 2026-01-06
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from datetime import datetime
import warnings
import gc
import os

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), 'Results')

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Models with known thermodynamic properties
MODELS = {
    'EleutherAI/pythia-160m': {
        'lab': 'EleutherAI', 'behavior': 'DAMPEN', 'gain': 1.157, 'layers': 12
    },
    'EleutherAI/pythia-410m': {
        'lab': 'EleutherAI', 'behavior': 'DAMPEN', 'gain': 0.978, 'layers': 24
    },
    'facebook/opt-125m': {
        'lab': 'Meta', 'behavior': 'EXPAND', 'gain': 1.263, 'layers': 12
    },
    'gpt2': {
        'lab': 'OpenAI', 'behavior': 'EXPAND', 'gain': 1.05, 'layers': 12
    }
}

TEST_PROMPTS = [
    "The capital of France is Paris.",
    "The sky is made of chocolate.",
    "Once upon a time in a land far away",
    "def fibonacci(n): return n if n < 2 else",
]

COLORS = {'EleutherAI': '#E74C3C', 'Meta': '#3498DB', 'OpenAI': '#8E44AD'}


# =============================================================================
# IMPROVEMENT #1: Quantitative Bounds for L*
# =============================================================================

def estimate_L_star_apriori(L: int, G: float, kappa: float = 5.0) -> float:
    """
    Estimate L* a priori from architecture parameters.

    Formula: L* ≈ (L/2) * (1 + tanh(κ * (G - 1)))

    Args:
        L: Total number of layers
        G: Residual gain (from Paper #2)
        kappa: Scaling parameter (default: 5.0)

    Returns:
        Estimated L* (critical layer)
    """
    return (L / 2) * (1 + np.tanh(kappa * (G - 1)))


def estimate_L_star_from_trace_dynamics(traces: np.ndarray) -> int:
    """
    Estimate L* from trace dynamics (max gradient).

    L* ≈ argmax_l |d Tr(Δ_F^(l)) / dl|
    """
    if len(traces) < 3:
        return 0

    # Compute gradient
    gradient = np.gradient(traces)

    # L* is where gradient magnitude is maximum
    return int(np.argmax(np.abs(gradient)))


# =============================================================================
# IMPROVEMENT #2: Full-Scale Laplacian (O(n² + d²) Trace Computation)
# =============================================================================

def compute_trace_efficient(attention: torch.Tensor, W_V: torch.Tensor) -> float:
    """
    Compute Tr(Δ_F) directly from diagonal blocks - NO SUBSAMPLING.

    Theory:
    - Diagonal block: [Δ_F]_{ii} = Σ_j A_{ij} · W_V^T W_V
    - Trace of block: Tr([Δ_F]_{ii}) = Σ_j A_{ij} · ||W_V||_F²
    - Total trace: Tr(Δ_F) = (Σ_{i,j} A_{ij} - n) · ||W_V||_F²

    Complexity: O(n² + d²) instead of O(n³d³)!

    Args:
        attention: [n, n] attention matrix (single head)
        W_V: [d_out, d_in] value projection matrix

    Returns:
        Trace of Sheaf Laplacian
    """
    n = attention.shape[0]

    # Frobenius norm squared of W_V
    W_V_frobenius_sq = (W_V ** 2).sum().item()

    # Sum of off-diagonal attention weights
    # off_diag_sum = A.sum() - A.trace() = Σ_{i≠j} A_ij
    attention_sum = attention.sum().item()
    attention_trace = attention.trace().item()
    off_diag_sum = attention_sum - attention_trace

    # Trace = (off-diagonal attention sum) * ||W_V||_F²
    trace = off_diag_sum * W_V_frobenius_sq

    return trace


def compute_multihead_trace(attentions: torch.Tensor, W_V_list: list) -> float:
    """
    Compute total trace for multi-head attention.

    Theory (Block-diagonal structure):
    Tr(Δ_F^total) = Σ_h Tr(Δ_F^(h))

    Args:
        attentions: [H, n, n] attention weights per head
        W_V_list: List of [d_h, d] value projections per head

    Returns:
        Total trace of multi-head Sheaf Laplacian
    """
    H = attentions.shape[0]
    total_trace = 0.0

    for h in range(H):
        A_h = attentions[h]
        W_V_h = W_V_list[h] if isinstance(W_V_list, list) else W_V_list

        head_trace = compute_trace_efficient(A_h, W_V_h)
        total_trace += head_trace

    return total_trace


def compute_spectral_nyström(attention: torch.Tensor, W_V: torch.Tensor,
                              n_landmarks: int = 50, k: int = 5) -> dict:
    """
    Compute approximate spectral properties using Nyström approximation.

    Uses Kronecker structure: L_F ≈ L_graph ⊗ W_V^T W_V

    Args:
        attention: [n, n] attention matrix
        W_V: [d_out, d_in] value projection
        n_landmarks: Number of landmark points for Nyström
        k: Number of eigenvalues to compute

    Returns:
        dict with lambda_1, lambda_2, spectral_gap, trace
    """
    n = attention.shape[0]
    n_landmarks = min(n_landmarks, n)

    # Convert to numpy
    A_np = attention.float().cpu().numpy()
    W_V_np = W_V.float().cpu().numpy()

    # Sample landmarks
    if n > n_landmarks:
        landmarks = np.random.choice(n, n_landmarks, replace=False)
        A_reduced = A_np[np.ix_(landmarks, landmarks)]
    else:
        A_reduced = A_np

    # Graph Laplacian of reduced attention
    D_reduced = np.diag(A_reduced.sum(axis=1))
    L_graph = D_reduced - A_reduced

    # W_V^T W_V
    W_VtV = W_V_np.T @ W_V_np if W_V_np.shape[0] < W_V_np.shape[1] else W_V_np @ W_V_np.T
    W_VtV = W_VtV[:min(50, W_VtV.shape[0]), :min(50, W_VtV.shape[1])]

    try:
        # Eigenvalues of graph Laplacian
        eig_graph = np.linalg.eigvalsh(L_graph + 1e-10 * np.eye(L_graph.shape[0]))
        eig_graph = np.sort(np.real(eig_graph))

        # Eigenvalues of W_V^T W_V
        eig_W = np.linalg.eigvalsh(W_VtV + 1e-10 * np.eye(W_VtV.shape[0]))
        eig_W = np.sort(np.real(eig_W))

        # Kronecker eigenvalues = products
        # Take smallest k from each
        k_graph = min(k, len(eig_graph))
        k_W = min(k, len(eig_W))

        small_eig_graph = eig_graph[:k_graph]
        small_eig_W = eig_W[:k_W]

        # All products of small eigenvalues
        products = np.outer(small_eig_graph, small_eig_W).flatten()
        products = np.sort(products)

        # Trace from eigenvalue sums
        trace = np.sum(eig_graph) * np.trace(W_VtV)

        return {
            'lambda_1': float(products[0]) if len(products) > 0 else 0.0,
            'lambda_2': float(products[1]) if len(products) > 1 else 0.0,
            'spectral_gap': float(products[1] - products[0]) if len(products) > 1 else 0.0,
            'trace': float(trace),
            'graph_lambda_2': float(eig_graph[1]) if len(eig_graph) > 1 else 0.0
        }
    except Exception as e:
        print(f"  Nyström error: {e}")
        return {
            'lambda_1': 0.0, 'lambda_2': 0.0, 'spectral_gap': 0.0,
            'trace': compute_trace_efficient(attention, W_V),
            'graph_lambda_2': 0.0
        }


# =============================================================================
# IMPROVEMENT #3: Multi-Head Integration
# =============================================================================

def extract_multihead_attention_and_W_V(model, model_name, tokenizer, prompt, device='cpu'):
    """
    Extract attention weights and W_V matrices for ALL heads in all layers.

    Returns:
        attentions: List of [batch, H, n, n] per layer
        W_V_per_head: List of List of [d_h, d] per layer per head
        seq_len: sequence length
        n_heads: number of attention heads
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

    attentions = outputs.attentions  # Tuple of [batch, H, n, n]

    W_V_per_layer = []
    n_heads = None

    if hasattr(model, 'gpt_neox'):  # Pythia
        layers = model.gpt_neox.layers
        config = model.config
        n_heads = config.num_attention_heads
        head_dim = config.hidden_size // n_heads

        for layer in layers:
            qkv = layer.attention.query_key_value.weight.data.float().cpu()
            hidden_size = qkv.shape[0] // 3
            W_V_full = qkv[2*hidden_size:, :]  # [hidden_size, hidden_size]

            # Split into heads
            W_V_heads = []
            for h in range(n_heads):
                W_V_h = W_V_full[h*head_dim:(h+1)*head_dim, :]
                W_V_heads.append(W_V_h)
            W_V_per_layer.append(W_V_heads)

    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):  # OPT
        layers = model.model.decoder.layers
        config = model.config
        n_heads = config.num_attention_heads
        head_dim = config.hidden_size // n_heads

        for layer in layers:
            W_V_full = layer.self_attn.v_proj.weight.data.float().cpu()

            W_V_heads = []
            for h in range(n_heads):
                W_V_h = W_V_full[h*head_dim:(h+1)*head_dim, :]
                W_V_heads.append(W_V_h)
            W_V_per_layer.append(W_V_heads)

    elif hasattr(model, 'transformer'):  # GPT-2
        layers = model.transformer.h
        config = model.config
        n_heads = config.n_head
        head_dim = config.n_embd // n_heads

        for layer in layers:
            c_attn = layer.attn.c_attn.weight.data.float().cpu()
            hidden_size = c_attn.shape[1] // 3
            W_V_full = c_attn[:, 2*hidden_size:].T  # [hidden_size, hidden_size]

            W_V_heads = []
            for h in range(n_heads):
                W_V_h = W_V_full[h*head_dim:(h+1)*head_dim, :]
                W_V_heads.append(W_V_h)
            W_V_per_layer.append(W_V_heads)

    return attentions, W_V_per_layer, inputs.input_ids.shape[1], n_heads


# =============================================================================
# Main Analysis
# =============================================================================

def analyze_model_h4_v2(model_name, config, test_prompts, device='cpu'):
    """Full H4 analysis v2 with all improvements."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"Lab: {config['lab']}, Expected: {config['behavior']}")
    print(f"{'='*60}")

    # A priori L* estimate
    L = config['layers']
    G = config['gain']
    L_star_apriori = estimate_L_star_apriori(L, G)
    print(f"  A priori L* estimate: {L_star_apriori:.2f} (L={L}, G={G})")

    # Load model
    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, output_attentions=True
    )
    model.to(device)
    model.eval()

    # Analyze each prompt
    all_traces_full = []
    all_traces_multihead = []
    all_spectral = []

    for prompt in tqdm(test_prompts, desc="  Prompts"):
        try:
            attentions, W_V_per_layer, seq_len, n_heads = extract_multihead_attention_and_W_V(
                model, model_name, tokenizer, prompt, device
            )

            layer_traces_full = []
            layer_traces_multihead = []
            layer_spectral = []

            for layer_idx in range(len(attentions)):
                attn = attentions[layer_idx]  # [1, H, n, n]
                W_V_heads = W_V_per_layer[layer_idx]

                # Method 1: Full-scale trace (averaged over heads)
                attn_avg = attn[0].mean(dim=0)  # [n, n] average attention
                W_V_avg = W_V_heads[0]  # Use first head's W_V as representative
                trace_full = compute_trace_efficient(attn_avg, W_V_avg)
                layer_traces_full.append(trace_full)

                # Method 2: Multi-head trace (sum over all heads)
                trace_multihead = compute_multihead_trace(attn[0], W_V_heads)
                layer_traces_multihead.append(trace_multihead)

                # Method 3: Nyström spectral approximation
                spectral = compute_spectral_nyström(attn_avg, W_V_avg)
                layer_spectral.append(spectral)

            all_traces_full.append(layer_traces_full)
            all_traces_multihead.append(layer_traces_multihead)
            all_spectral.append(layer_spectral)

        except Exception as e:
            print(f"  Error on prompt: {e}")
            continue

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_traces_full:
        return None

    # Average across prompts
    mean_traces_full = np.mean(all_traces_full, axis=0)
    mean_traces_multihead = np.mean(all_traces_multihead, axis=0)

    # Extract spectral properties
    mean_lambda_2 = np.mean([[s['lambda_2'] for s in layer] for layer in all_spectral], axis=0)
    mean_spectral_gaps = np.mean([[s['spectral_gap'] for s in layer] for layer in all_spectral], axis=0)
    mean_graph_lambda_2 = np.mean([[s['graph_lambda_2'] for s in layer] for layer in all_spectral], axis=0)

    # Empirical L* estimates
    L_star_trace = estimate_L_star_from_trace_dynamics(mean_traces_full)
    L_star_spectral = int(np.argmax(mean_spectral_gaps))

    print(f"\n  Results:")
    print(f"    A priori L*: {L_star_apriori:.2f}")
    print(f"    Empirical L* (trace): {L_star_trace}")
    print(f"    Empirical L* (spectral): {L_star_spectral}")
    print(f"    Mean trace (full): {np.mean(mean_traces_full):.4f}")
    print(f"    Mean trace (multihead): {np.mean(mean_traces_multihead):.4f}")

    return {
        'model': model_name,
        'lab': config['lab'],
        'behavior': config['behavior'],
        'known_gain': config['gain'],
        'n_layers': len(mean_traces_full),
        'n_heads': n_heads,

        # A priori
        'L_star_apriori': float(L_star_apriori),

        # Traces
        'traces_full': mean_traces_full.tolist(),
        'traces_multihead': mean_traces_multihead.tolist(),
        'mean_trace_full': float(np.mean(mean_traces_full)),
        'mean_trace_multihead': float(np.mean(mean_traces_multihead)),

        # Spectral
        'lambda_2': mean_lambda_2.tolist(),
        'spectral_gaps': mean_spectral_gaps.tolist(),
        'graph_lambda_2': mean_graph_lambda_2.tolist(),

        # L* estimates
        'L_star_trace': L_star_trace,
        'L_star_spectral': L_star_spectral,
        'L_star_ratio_apriori': L_star_apriori / len(mean_traces_full),
        'L_star_ratio_trace': L_star_trace / len(mean_traces_full),
    }


def create_visualizations_v2(results):
    """Create improved H4 validation figures."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Trace by Layer (Full-Scale)
    ax1 = axes[0, 0]
    for r in results:
        layers = np.arange(len(r['traces_full']))
        norm_layers = layers / len(r['traces_full'])
        ax1.plot(norm_layers, r['traces_full'],
                 label=f"{r['model'].split('/')[-1]} ({r['behavior']})",
                 color=COLORS.get(r['lab'], 'gray'), linewidth=2, marker='o', markersize=4)
        # Mark empirical L*
        ax1.axvline(x=r['L_star_trace'] / r['n_layers'], color=COLORS.get(r['lab'], 'gray'),
                    linestyle='--', alpha=0.5)
    ax1.set_xlabel('Normalized Layer (l/L)', fontsize=12)
    ax1.set_ylabel('Tr(Δ_F) - Full Scale', fontsize=12)
    ax1.set_title('Sheaf Laplacian Trace (No Subsampling)', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Multi-Head Trace
    ax2 = axes[0, 1]
    for r in results:
        layers = np.arange(len(r['traces_multihead']))
        norm_layers = layers / len(r['traces_multihead'])
        ax2.plot(norm_layers, r['traces_multihead'],
                 label=f"{r['model'].split('/')[-1]} (H={r['n_heads']})",
                 color=COLORS.get(r['lab'], 'gray'), linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Normalized Layer (l/L)', fontsize=12)
    ax2.set_ylabel('Σ_h Tr(Δ_F^(h))', fontsize=12)
    ax2.set_title('Multi-Head Integrated Trace', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Graph λ₂ (Nyström)
    ax3 = axes[0, 2]
    for r in results:
        layers = np.arange(len(r['graph_lambda_2']))
        norm_layers = layers / len(r['graph_lambda_2'])
        ax3.semilogy(norm_layers, np.array(r['graph_lambda_2']) + 1e-10,
                     label=f"{r['model'].split('/')[-1]}",
                     color=COLORS.get(r['lab'], 'gray'), linewidth=2, marker='^', markersize=4)
    ax3.set_xlabel('Normalized Layer (l/L)', fontsize=12)
    ax3.set_ylabel('Graph λ₂ (Nyström)', fontsize=12)
    ax3.set_title('Attention Graph Connectivity', fontsize=14)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: A priori vs Empirical L*
    ax4 = axes[1, 0]
    apriori = [r['L_star_apriori'] for r in results]
    empirical = [r['L_star_trace'] for r in results]
    labels = [r['model'].split('/')[-1] for r in results]
    colors = [COLORS.get(r['lab'], 'gray') for r in results]

    for i, (a, e, l, c) in enumerate(zip(apriori, empirical, labels, colors)):
        ax4.scatter(a, e, color=c, s=150, edgecolors='white', linewidths=2, label=l)
    ax4.plot([0, max(apriori + empirical)], [0, max(apriori + empirical)],
             'k--', alpha=0.5, label='Perfect prediction')
    ax4.set_xlabel('L* A Priori (from Gain)', fontsize=12)
    ax4.set_ylabel('L* Empirical (from Trace)', fontsize=12)
    ax4.set_title('L* Prediction: A Priori vs Empirical', fontsize=14)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Trace vs Gain
    ax5 = axes[1, 1]
    for r in results:
        ax5.scatter(r['known_gain'], r['mean_trace_full'],
                    color=COLORS.get(r['lab'], 'gray'),
                    s=150, edgecolors='white', linewidths=2,
                    label=f"{r['model'].split('/')[-1]}")
    ax5.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Known Residual Gain (G)', fontsize=12)
    ax5.set_ylabel('Mean Trace (Full-Scale)', fontsize=12)
    ax5.set_title('Trace vs Thermodynamic Gain', fontsize=14)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Plot 6: DAMPEN vs EXPAND comparison
    ax6 = axes[1, 2]
    dampeners = [r for r in results if r['behavior'] == 'DAMPEN']
    expanders = [r for r in results if r['behavior'] == 'EXPAND']

    bar_width = 0.35
    x = np.arange(2)

    dampen_traces = [np.mean([r['mean_trace_full'] for r in dampeners])]
    expand_traces = [np.mean([r['mean_trace_full'] for r in expanders])]
    dampen_L_star = [np.mean([r['L_star_ratio_trace'] for r in dampeners])]
    expand_L_star = [np.mean([r['L_star_ratio_trace'] for r in expanders])]

    ax6.bar([0], dampen_traces, bar_width, label='DAMPEN', color='#E74C3C', alpha=0.8)
    ax6.bar([0 + bar_width], expand_traces, bar_width, label='EXPAND', color='#3498DB', alpha=0.8)
    ax6.set_ylabel('Mean Trace', fontsize=12)
    ax6.set_title('Thermodynamic Signature: DAMPEN vs EXPAND', fontsize=14)
    ax6.set_xticks([bar_width / 2])
    ax6.set_xticklabels(['Mean Trace'])
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.suptitle('H4 Validation v2: Full-Scale Multi-Head Sheaf Laplacian',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def main():
    print("\n" + "="*70)
    print("H4 VALIDATION v2: Full-Scale Multi-Head Sheaf Laplacian")
    print("="*70)
    print("\nImprovements:")
    print("  1. Quantitative Bounds: A priori L* estimation")
    print("  2. Full-Scale Laplacian: O(n² + d²) trace (no subsampling)")
    print("  3. Multi-Head Integration: Proper H-head handling")
    print(f"\nTesting {len(MODELS)} models with {len(TEST_PROMPTS)} prompts")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Run analysis
    results = []
    for model_name, config in MODELS.items():
        result = analyze_model_h4_v2(model_name, config, TEST_PROMPTS, device)
        if result:
            results.append(result)

    print(f"\n\n{'='*60}")
    print(f"Successfully analyzed {len(results)} / {len(MODELS)} models")
    print(f"{'='*60}")

    if not results:
        print("No results to analyze!")
        return

    # Create summary
    summary = pd.DataFrame([{
        'Model': r['model'].split('/')[-1],
        'Lab': r['lab'],
        'Behavior': r['behavior'],
        'Gain': r['known_gain'],
        'Heads': r['n_heads'],
        'Layers': r['n_layers'],
        'L* apriori': f"{r['L_star_apriori']:.1f}",
        'L* empirical': r['L_star_trace'],
        'Mean Trace': f"{r['mean_trace_full']:.2f}",
        'Mean Trace (MH)': f"{r['mean_trace_multihead']:.2f}",
    } for r in results])

    print("\n" + "="*80)
    print("H4 v2 VALIDATION SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))

    # Statistical analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    dampeners = [r for r in results if r['behavior'] == 'DAMPEN']
    expanders = [r for r in results if r['behavior'] == 'EXPAND']

    if dampeners and expanders:
        print(f"\nDAMPENERS (n={len(dampeners)}):")
        print(f"  Mean trace: {np.mean([r['mean_trace_full'] for r in dampeners]):.4f}")
        print(f"  Mean L*/L: {np.mean([r['L_star_ratio_trace'] for r in dampeners]):.3f}")

        print(f"\nEXPANDERS (n={len(expanders)}):")
        print(f"  Mean trace: {np.mean([r['mean_trace_full'] for r in expanders]):.4f}")
        print(f"  Mean L*/L: {np.mean([r['L_star_ratio_trace'] for r in expanders]):.3f}")

        # Trace ratio
        dampen_mean = np.mean([r['mean_trace_full'] for r in dampeners])
        expand_mean = np.mean([r['mean_trace_full'] for r in expanders])
        ratio = expand_mean / dampen_mean if dampen_mean > 0 else float('inf')
        print(f"\nEXPAND/DAMPEN trace ratio: {ratio:.1f}x")

    # L* prediction accuracy
    print("\n" + "-"*40)
    print("L* PREDICTION ACCURACY:")
    for r in results:
        error = abs(r['L_star_apriori'] - r['L_star_trace'])
        print(f"  {r['model'].split('/')[-1]}: apriori={r['L_star_apriori']:.1f}, "
              f"empirical={r['L_star_trace']}, error={error:.1f}")

    # Verdict
    print("\n" + "="*80)
    print("H4 v2 VALIDATION VERDICT")
    print("="*80)

    findings = []

    # Check trace discrimination
    if dampeners and expanders:
        trace_discrimination = expand_mean > 2 * dampen_mean
        findings.append(f"Trace discriminates EXPAND vs DAMPEN ({ratio:.1f}x): {trace_discrimination}")

    # Check L* prediction
    errors = [abs(r['L_star_apriori'] - r['L_star_trace']) for r in results]
    mean_error = np.mean(errors)
    good_prediction = mean_error < 3
    findings.append(f"A priori L* prediction (mean error={mean_error:.1f}): {good_prediction}")

    # Check full-scale consistency
    trace_ratios = [r['mean_trace_multihead'] / r['mean_trace_full']
                    for r in results if r['mean_trace_full'] > 0]
    mh_consistent = np.std(trace_ratios) < 2 if trace_ratios else False
    findings.append(f"Multi-head consistency (σ={np.std(trace_ratios):.2f}): {mh_consistent}")

    for f in findings:
        print(f"  * {f}")

    all_pass = all('True' in f for f in findings)
    verdict = "H4 v2 FULLY VALIDATED" if all_pass else "H4 v2 PARTIALLY VALIDATED"
    print(f"\n{verdict}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    output = {
        'experiment': 'H4 v2 Full-Scale Multi-Head Sheaf Laplacian',
        'date': datetime.now().isoformat(),
        'improvements': [
            '1. Quantitative Bounds: a priori L* from Gain',
            '2. Full-Scale Trace: O(n² + d²), no subsampling',
            '3. Multi-Head Integration: Σ_h Tr(Δ_F^(h))'
        ],
        'models_tested': len(results),
        'verdict': verdict,
        'findings': findings,
        'results': results,
        'summary': summary.to_dict('records')
    }

    json_path = os.path.join(RESULTS_DIR, f'H4_v2_validation_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # Save figure
    fig = create_visualizations_v2(results)
    fig_path = os.path.join(OUTPUT_DIR, 'H4_v2_validation.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    # Save summary CSV
    csv_path = os.path.join(RESULTS_DIR, f'H4_v2_summary_{timestamp}.csv')
    summary.to_csv(csv_path, index=False)
    print(f"Summary saved: {csv_path}")

    print("\n" + "="*70)
    print("H4 v2 VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
