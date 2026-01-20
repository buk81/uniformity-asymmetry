"""
Paper 4: Alignment Robustness - Shared Utilities

This module provides shared functions for:
- Specialization Index (SI) computation
- Perplexity (PPL) measurement
- Attention pattern analysis
"""

from .metrics import compute_si, compute_ppl, compute_attention_entropy

__all__ = ['compute_si', 'compute_ppl', 'compute_attention_entropy']
__version__ = '1.0.0'
