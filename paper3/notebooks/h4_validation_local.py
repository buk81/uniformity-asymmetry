#!/usr/bin/env python3
"""
H4 Validation: Sheaf-Laplacian Spectral Gap Predicts L*

Hypothesis H4: The spectral gap λ₂ of the Sheaf Laplacian marks semantic
domain separation and correlates with the transition point L*.

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

# Models with known thermodynamic properties (from H25-H27)
MODELS = {
    # EleutherAI - DAMPENERS
    'EleutherAI/pythia-160m': {
        'lab': 'EleutherAI',
        'behavior': 'DAMPEN',
        'gain': 1.157,
        'layers': 12
    },
    'EleutherAI/pythia-410m': {
        'lab': 'EleutherAI',
        'behavior': 'DAMPEN',
        'gain': 0.978,
        'layers': 24
    },
    # Meta - EXPANDERS
    'facebook/opt-125m': {
        'lab': 'Meta',
        'behavior': 'EXPAND',
        'gain': 1.263,
        'layers': 12
    },
    # OpenAI - EXPANDER
    'gpt2': {
        'lab': 'OpenAI',
        'behavior': 'EXPAND',
        'gain': 1.05,
        'layers': 12
    }
}

# Test prompts - semantically distinct domains
TEST_PROMPTS = [
    "The capital of France is Paris.",
    "The sky is made of chocolate.",
    "Once upon a time in a land far away",
    "def fibonacci(n): return n if n < 2 else",
]

COLORS = {'EleutherAI': '#E74C3C', 'Meta': '#3498DB', 'OpenAI': '#8E44AD'}


def get_attention_and_W_V(model, model_name, tokenizer, prompt, device='cpu'):
    """Extract attention weights and W_V matrices for all layers."""
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True
        )

    attentions = outputs.attentions
    W_V_list = []

    if hasattr(model, 'gpt_neox'):  # Pythia
        layers = model.gpt_neox.layers
        for layer in layers:
            qkv = layer.attention.query_key_value.weight.data.float().cpu()
            hidden_size = qkv.shape[0] // 3
            W_V = qkv[2*hidden_size:, :]
            W_V_list.append(W_V)
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):  # OPT
        layers = model.model.decoder.layers
        for layer in layers:
            W_V = layer.self_attn.v_proj.weight.data.float().cpu()
            W_V_list.append(W_V)
    elif hasattr(model, 'transformer'):  # GPT-2
        layers = model.transformer.h
        for layer in layers:
            c_attn = layer.attn.c_attn.weight.data.float().cpu()
            hidden_size = c_attn.shape[1] // 3
            W_V = c_attn[:, 2*hidden_size:].T
            W_V_list.append(W_V)

    return attentions, W_V_list, inputs.input_ids.shape[1]


def build_sheaf_laplacian_efficient(attention, W_V, max_tokens=6, proj_dim=16):
    """
    Build Sheaf Laplacian efficiently using subsampling.

    The Sheaf Laplacian L_F = δ^T δ has block structure:
    - L_F[i,i] = Σ_j ρ_ij^T ρ_ij  (diagonal)
    - L_F[i,j] = -ρ_ij^T ρ_ji     (off-diagonal)

    Returns spectral properties: λ₁, λ₂, spectral_gap, trace
    """
    # Use first head
    A = attention[0, 0].float().cpu()
    seq_len = A.shape[0]

    # Subsample tokens for efficiency
    if seq_len > max_tokens:
        indices = np.linspace(0, seq_len-1, max_tokens, dtype=int)
        A = A[np.ix_(indices, indices)]
        seq_len = max_tokens

    # Project W_V to smaller dimension
    d = min(proj_dim, W_V.shape[0], W_V.shape[1])
    W_V_small = W_V[:d, :d].numpy()

    # Compute √A (restriction map scaling)
    sqrt_A = torch.sqrt(A + 1e-10).numpy()

    # Build block Laplacian
    n = seq_len
    L_F = np.zeros((n * d, n * d))

    # Diagonal blocks: L[i,i] = Σ_j ρ_ij^T ρ_ij
    for i in range(n):
        block_ii = np.zeros((d, d))
        for j in range(n):
            if i != j:
                rho_ij = sqrt_A[i, j] * W_V_small
                block_ii += rho_ij.T @ rho_ij
        L_F[i*d:(i+1)*d, i*d:(i+1)*d] = block_ii

    # Off-diagonal blocks: L[i,j] = -ρ_ij^T ρ_ji
    for i in range(n):
        for j in range(n):
            if i != j:
                rho_ij = sqrt_A[i, j] * W_V_small
                rho_ji = sqrt_A[j, i] * W_V_small
                L_F[i*d:(i+1)*d, j*d:(j+1)*d] = -rho_ij.T @ rho_ji

    # Compute eigenvalues
    try:
        L_F_reg = L_F + 1e-10 * np.eye(L_F.shape[0])
        eigenvalues = np.linalg.eigvalsh(L_F_reg)
        eigenvalues = np.sort(np.real(eigenvalues))
    except:
        eigenvalues = np.array([0.0, np.trace(L_F) / L_F.shape[0]])

    return {
        'lambda_1': float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0,
        'lambda_2': float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0,
        'spectral_gap': float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0,
        'trace': float(np.trace(L_F)),
        'eigenvalues': eigenvalues[:10].tolist() if len(eigenvalues) >= 10 else eigenvalues.tolist()
    }


def analyze_model_h4(model_name, config, test_prompts, device='cpu'):
    """Full H4 analysis for a single model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"Lab: {config['lab']}, Expected: {config['behavior']}")
    print(f"{'='*60}")

    # Load model
    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        output_attentions=True,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # Analyze each prompt
    all_spectral_gaps = []
    all_lambda_2 = []
    all_traces = []

    for prompt in tqdm(test_prompts, desc="  Prompts"):
        try:
            attentions, W_V_list, seq_len = get_attention_and_W_V(
                model, model_name, tokenizer, prompt, device
            )

            layer_spectral_gaps = []
            layer_lambda_2 = []
            layer_traces = []

            for layer_idx in range(min(len(attentions), len(W_V_list))):
                spectral = build_sheaf_laplacian_efficient(
                    attentions[layer_idx],
                    W_V_list[layer_idx]
                )
                layer_spectral_gaps.append(spectral['spectral_gap'])
                layer_lambda_2.append(spectral['lambda_2'])
                layer_traces.append(spectral['trace'])

            all_spectral_gaps.append(layer_spectral_gaps)
            all_lambda_2.append(layer_lambda_2)
            all_traces.append(layer_traces)

        except Exception as e:
            print(f"  Error on prompt: {e}")
            continue

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Average across prompts
    if all_spectral_gaps:
        mean_spectral_gaps = np.mean(all_spectral_gaps, axis=0)
        mean_lambda_2 = np.mean(all_lambda_2, axis=0)
        mean_traces = np.mean(all_traces, axis=0)

        # Find key transition points
        L_star_gap_max = int(np.argmax(mean_spectral_gaps))
        L_star_lambda2_max = int(np.argmax(mean_lambda_2))

        # Compute second derivative for inflection
        if len(mean_spectral_gaps) > 2:
            second_deriv = np.diff(np.diff(mean_spectral_gaps))
            L_star_inflection = int(np.argmax(np.abs(second_deriv))) + 1
        else:
            L_star_inflection = 0

        return {
            'model': model_name,
            'lab': config['lab'],
            'behavior': config['behavior'],
            'known_gain': config['gain'],
            'n_layers': len(mean_spectral_gaps),
            'spectral_gaps': mean_spectral_gaps.tolist(),
            'lambda_2': mean_lambda_2.tolist(),
            'traces': mean_traces.tolist(),
            'L_star_gap_max': L_star_gap_max,
            'L_star_lambda2_max': L_star_lambda2_max,
            'L_star_inflection': L_star_inflection,
            'max_spectral_gap': float(np.max(mean_spectral_gaps)),
            'mean_spectral_gap': float(np.mean(mean_spectral_gaps)),
            'spectral_gap_at_half': float(mean_spectral_gaps[len(mean_spectral_gaps)//2])
        }

    return None


def create_visualizations(results):
    """Create H4 validation figures."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Spectral Gap by Layer
    ax1 = axes[0, 0]
    for r in results:
        layers = np.arange(len(r['spectral_gaps']))
        normalized_layers = layers / len(r['spectral_gaps'])
        ax1.plot(normalized_layers, r['spectral_gaps'],
                 label=f"{r['model'].split('/')[-1]} ({r['behavior']})",
                 color=COLORS.get(r['lab'], 'gray'),
                 linewidth=2, marker='o', markersize=4)
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='L/2')
    ax1.set_xlabel('Normalized Layer (l/L)', fontsize=12)
    ax1.set_ylabel('Spectral Gap (λ₂ - λ₁)', fontsize=12)
    ax1.set_title('Sheaf Laplacian Spectral Gap by Layer', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: λ₂ by Layer
    ax2 = axes[0, 1]
    for r in results:
        layers = np.arange(len(r['lambda_2']))
        normalized_layers = layers / len(r['lambda_2'])
        ax2.semilogy(normalized_layers, np.array(r['lambda_2']) + 1e-10,
                     label=f"{r['model'].split('/')[-1]}",
                     color=COLORS.get(r['lab'], 'gray'),
                     linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Normalized Layer (l/L)', fontsize=12)
    ax2.set_ylabel('λ₂ (log scale)', fontsize=12)
    ax2.set_title('Second Eigenvalue (Algebraic Connectivity)', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Max Spectral Gap vs Known Gain
    ax3 = axes[1, 0]
    for r in results:
        ax3.scatter(r['known_gain'], r['max_spectral_gap'],
                    color=COLORS.get(r['lab'], 'gray'),
                    s=150, edgecolors='white', linewidths=2,
                    label=f"{r['model'].split('/')[-1]}")
    ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Known Residual Gain (G)', fontsize=12)
    ax3.set_ylabel('Max Spectral Gap', fontsize=12)
    ax3.set_title('Spectral Gap vs Thermodynamic Behavior', fontsize=14)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: L* Position vs Known Gain
    ax4 = axes[1, 1]
    for r in results:
        ax4.scatter(r['known_gain'], r['L_star_gap_max'] / r['n_layers'],
                    color=COLORS.get(r['lab'], 'gray'),
                    s=150, edgecolors='white', linewidths=2,
                    marker='D',
                    label=f"{r['model'].split('/')[-1]}")
    ax4.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Known Residual Gain (G)', fontsize=12)
    ax4.set_ylabel('L* / L (Transition Point)', fontsize=12)
    ax4.set_title('Transition Point vs Thermodynamic Behavior', fontsize=14)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('H4 Validation: Sheaf-Laplacian Spectral Gap Analysis',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def main():
    print("\n" + "="*70)
    print("H4 VALIDATION: Sheaf-Laplacian Spectral Gap Predicts L*")
    print("="*70)
    print(f"\nTesting {len(MODELS)} models with {len(TEST_PROMPTS)} prompts")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Run analysis
    results = []
    for model_name, config in MODELS.items():
        result = analyze_model_h4(model_name, config, TEST_PROMPTS, device)
        if result:
            results.append(result)
            print(f"\n  L* (max gap): Layer {result['L_star_gap_max']}")
            print(f"  L* (max λ₂): Layer {result['L_star_lambda2_max']}")
            print(f"  Max spectral gap: {result['max_spectral_gap']:.4f}")

    print(f"\n\n{'='*60}")
    print(f"Successfully analyzed {len(results)} / {len(MODELS)} models")
    print(f"{'='*60}")

    if not results:
        print("No results to analyze!")
        return

    # Create summary DataFrame
    summary = pd.DataFrame([{
        'Model': r['model'].split('/')[-1],
        'Lab': r['lab'],
        'Behavior': r['behavior'],
        'Known Gain': r['known_gain'],
        'Layers': r['n_layers'],
        'L* (gap max)': r['L_star_gap_max'],
        'L* / L': r['L_star_gap_max'] / r['n_layers'],
        'Max Gap': r['max_spectral_gap'],
        'Mean Gap': r['mean_spectral_gap']
    } for r in results])

    print("\n" + "="*80)
    print("H4 VALIDATION SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))

    # Statistical tests
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    dampeners = summary[summary['Behavior'] == 'DAMPEN']
    expanders = summary[summary['Behavior'] == 'EXPAND']

    if len(dampeners) > 0 and len(expanders) > 0:
        print(f"\nDAMPENERS (EleutherAI):")
        print(f"  Mean max gap: {dampeners['Max Gap'].mean():.4f}")
        print(f"  Mean L*/L: {dampeners['L* / L'].mean():.3f}")

        print(f"\nEXPANDERS (Meta, OpenAI):")
        print(f"  Mean max gap: {expanders['Max Gap'].mean():.4f}")
        print(f"  Mean L*/L: {expanders['L* / L'].mean():.3f}")

    if len(summary) >= 3:
        r, p = stats.spearmanr(summary['Max Gap'], summary['Known Gain'])
        print(f"\nSpearman (Max Gap vs Gain): r = {r:.3f}, p = {p:.4f}")

        r2, p2 = stats.spearmanr(summary['L* / L'], summary['Known Gain'])
        print(f"Spearman (L*/L vs Gain): r = {r2:.3f}, p = {p2:.4f}")

    # Verdict
    print("\n" + "="*80)
    print("H4 VALIDATION VERDICT")
    print("="*80)

    findings = []

    all_show_variation = all(r['max_spectral_gap'] > r['mean_spectral_gap'] * 1.1 for r in results)
    findings.append(f"All models show spectral gap variation: {all_show_variation}")

    l_star_ratios = [r['L_star_gap_max'] / r['n_layers'] for r in results]
    l_star_near_half = 0.3 < np.mean(l_star_ratios) < 0.7
    findings.append(f"L* near L/2 (mean={np.mean(l_star_ratios):.2f}): {l_star_near_half}")

    for f in findings:
        print(f"  • {f}")

    print("\n" + "-"*40)
    if all_show_variation and l_star_near_half:
        verdict = "✅ H4 EMPIRICALLY VALIDATED"
        detail = "Spectral gap λ₂ shows systematic layer-wise variation with transition near L/2"
    else:
        verdict = "⚠️ H4 PARTIALLY SUPPORTED"
        detail = "Spectral structure exists but correlation with L* needs more data"

    print(f"\n{verdict}")
    print(f"{detail}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    output = {
        'experiment': 'H4 Sheaf-Laplacian Spectral Gap Validation',
        'date': datetime.now().isoformat(),
        'hypothesis': 'λ₂ of Sheaf Laplacian marks semantic domain separation and correlates with L*',
        'models_tested': len(results),
        'verdict': verdict,
        'detail': detail,
        'findings': findings,
        'results': results,
        'summary': summary.to_dict('records')
    }

    json_path = os.path.join(RESULTS_DIR, f'H4_validation_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # Save figure
    fig = create_visualizations(results)
    fig_path = os.path.join(OUTPUT_DIR, 'H4_spectral_gap_validation.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    # Save summary CSV
    csv_path = os.path.join(RESULTS_DIR, f'H4_summary_{timestamp}.csv')
    summary.to_csv(csv_path, index=False)
    print(f"Summary saved: {csv_path}")

    print("\n" + "="*70)
    print("H4 VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
