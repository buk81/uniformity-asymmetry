"""
Shared metrics computation for Paper 4: Alignment Robustness.

This module provides the core metrics used across all experiments:
- Specialization Index (SI): Measures attention head specialization
- Perplexity (PPL): Standard language model perplexity
- Attention Entropy: Measures attention distribution uniformity

All functions are designed to work with HuggingFace transformers models.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def compute_attention_entropy(attention_weights: "torch.Tensor") -> float:
    """
    Compute the entropy of attention weights.

    H(A) = -sum(A * log(A + eps))

    Parameters
    ----------
    attention_weights : torch.Tensor
        Attention weights of shape (batch, heads, seq_len, seq_len)

    Returns
    -------
    float
        Average entropy across all heads
    """
    if not HAS_TORCH:
        raise ImportError("torch required for attention entropy computation")

    eps = 1e-10
    # Ensure attention weights sum to 1 along last dimension
    attn = attention_weights.clamp(min=eps)

    # Compute entropy: -sum(p * log(p))
    entropy = -(attn * torch.log(attn)).sum(dim=-1)

    # Average across batch, heads, and positions
    return entropy.mean().item()


def compute_si(
    model: "torch.nn.Module",
    tokenizer: Any,
    prompt: str,
    layer_idx: int = -1,
    device: str = "cuda",
    max_length: int = 128
) -> float:
    """
    Compute Specialization Index (SI) for a given prompt.

    SI measures how specialized attention heads are, defined as:
    SI = 1 - (H_actual / H_max)

    where H is entropy. Higher SI = more specialized attention.

    Parameters
    ----------
    model : torch.nn.Module
        HuggingFace transformer model
    tokenizer : Any
        Model tokenizer
    prompt : str
        Input prompt text
    layer_idx : int
        Layer index to analyze (-1 for last layer)
    device : str
        Device to run on ('cuda' or 'cpu')
    max_length : int
        Maximum sequence length

    Returns
    -------
    float
        Specialization Index in range [0, 1]
    """
    if not HAS_TORCH:
        raise ImportError("torch required for SI computation")

    model.eval()

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass with attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get attention weights for specified layer
    attentions = outputs.attentions
    if layer_idx < 0:
        layer_idx = len(attentions) + layer_idx
    attn_weights = attentions[layer_idx]

    # Compute entropy
    H_actual = compute_attention_entropy(attn_weights)

    # Maximum entropy (uniform distribution over seq_len)
    seq_len = attn_weights.shape[-1]
    H_max = np.log(seq_len)

    # SI = 1 - normalized entropy
    si = 1.0 - (H_actual / H_max)

    return float(si)


def compute_si_all_layers(
    model: "torch.nn.Module",
    tokenizer: Any,
    prompt: str,
    device: str = "cuda",
    max_length: int = 128
) -> List[float]:
    """
    Compute SI for all layers.

    Parameters
    ----------
    model : torch.nn.Module
        HuggingFace transformer model
    tokenizer : Any
        Model tokenizer
    prompt : str
        Input prompt text
    device : str
        Device to run on
    max_length : int
        Maximum sequence length

    Returns
    -------
    List[float]
        SI values for each layer
    """
    if not HAS_TORCH:
        raise ImportError("torch required for SI computation")

    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions
    seq_len = attentions[0].shape[-1]
    H_max = np.log(seq_len)

    si_values = []
    for attn_weights in attentions:
        H_actual = compute_attention_entropy(attn_weights)
        si = 1.0 - (H_actual / H_max)
        si_values.append(float(si))

    return si_values


def compute_ppl(
    model: "torch.nn.Module",
    tokenizer: Any,
    text: str,
    device: str = "cuda",
    max_length: int = 128
) -> float:
    """
    Compute perplexity for a given text.

    PPL = exp(average_negative_log_likelihood)

    Parameters
    ----------
    model : torch.nn.Module
        HuggingFace transformer model
    tokenizer : Any
        Model tokenizer
    text : str
        Input text
    device : str
        Device to run on
    max_length : int
        Maximum sequence length

    Returns
    -------
    float
        Perplexity value
    """
    if not HAS_TORCH:
        raise ImportError("torch required for PPL computation")

    model.eval()

    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])

    # Loss is average negative log-likelihood
    loss = outputs.loss.item()
    ppl = np.exp(loss)

    return float(ppl)


def compute_delta_si(
    si_baseline: float,
    si_perturbed: float
) -> Tuple[float, float]:
    """
    Compute delta SI and percentage change.

    Parameters
    ----------
    si_baseline : float
        SI value before perturbation
    si_perturbed : float
        SI value after perturbation

    Returns
    -------
    Tuple[float, float]
        (delta_si, percentage_change)
    """
    delta = si_perturbed - si_baseline
    pct_change = (delta / si_baseline) * 100 if si_baseline != 0 else 0

    return delta, pct_change


def compute_ppl_ratio(
    ppl_instruct: float,
    ppl_base: float
) -> float:
    """
    Compute PPL ratio (instruct / base).

    This ratio indicates alignment overhead:
    - Ratio > 1: Alignment increases perplexity (common for instruct models)
    - Ratio >> 1: High alignment overhead (potential fragility indicator)

    Parameters
    ----------
    ppl_instruct : float
        Perplexity of instruct/aligned model
    ppl_base : float
        Perplexity of base model

    Returns
    -------
    float
        PPL ratio
    """
    if ppl_base == 0:
        return float('inf')
    return ppl_instruct / ppl_base


def analyze_attention_pattern(
    attention_weights: "torch.Tensor"
) -> Dict[str, float]:
    """
    Analyze attention pattern statistics.

    Parameters
    ----------
    attention_weights : torch.Tensor
        Attention weights of shape (batch, heads, seq_len, seq_len)

    Returns
    -------
    Dict[str, float]
        Dictionary with pattern statistics
    """
    if not HAS_TORCH:
        raise ImportError("torch required for attention analysis")

    # Get various statistics
    attn = attention_weights.squeeze(0)  # Remove batch dim

    # Max attention per head (sink detection)
    max_attn = attn.max(dim=-1).values.max(dim=-1).values  # (heads,)

    # Attention concentration on first token (sink pattern)
    first_token_attn = attn[:, :, 0].mean(dim=-1)  # (heads,)

    # Diagonal attention (local pattern)
    diag_mask = torch.eye(attn.shape[-1], device=attn.device, dtype=torch.bool)
    diag_attn = attn[:, diag_mask].mean(dim=-1)  # (heads,)

    return {
        "max_attention_mean": max_attn.mean().item(),
        "max_attention_std": max_attn.std().item(),
        "first_token_attention_mean": first_token_attn.mean().item(),
        "diagonal_attention_mean": diag_attn.mean().item(),
        "n_heads": int(attn.shape[0]),
        "seq_len": int(attn.shape[-1])
    }


# Convenience function for batch processing
def compute_si_batch(
    model: "torch.nn.Module",
    tokenizer: Any,
    prompts: List[str],
    device: str = "cuda",
    max_length: int = 128
) -> Dict[str, Any]:
    """
    Compute SI for multiple prompts.

    Parameters
    ----------
    model : torch.nn.Module
        HuggingFace transformer model
    tokenizer : Any
        Model tokenizer
    prompts : List[str]
        List of prompt texts
    device : str
        Device to run on
    max_length : int
        Maximum sequence length

    Returns
    -------
    Dict[str, Any]
        Dictionary with per-prompt and aggregate SI values
    """
    si_values = []
    for prompt in prompts:
        si = compute_si(model, tokenizer, prompt, device=device, max_length=max_length)
        si_values.append(si)

    return {
        "si_per_prompt": si_values,
        "si_mean": float(np.mean(si_values)),
        "si_std": float(np.std(si_values)),
        "si_min": float(np.min(si_values)),
        "si_max": float(np.max(si_values)),
        "n_prompts": len(prompts)
    }
