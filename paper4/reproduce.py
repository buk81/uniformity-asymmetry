#!/usr/bin/env python3
"""
CLI for reproducing Paper 4 experiments.

Currently supports:
- E11: Territorial Collapse (minimal reproduction)
- E04: Heritage Twin Test (minimal reproduction)

Note: Notebooks remain canonical for full sweep reproduction.
This CLI provides a quick verification path for core experiments.

Usage:
    python reproduce.py --experiment E11 --model mistral --seed 42
    python reproduce.py --experiment E04 --model llama31 --seed 42
    python reproduce.py --list-models
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    # E11 Territorial Collapse models
    "mistral": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "architecture": "GQA",
        "kv_heads": "8"
    },
    "llama2": {
        "hf_id": "meta-llama/Llama-2-7b-chat-hf",
        "architecture": "MHA",
        "kv_heads": "32"
    },
    "llama31": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "architecture": "GQA",
        "kv_heads": "8"
    },
    "yi15": {
        "hf_id": "01-ai/Yi-1.5-9B-Chat",
        "architecture": "GQA",
        "kv_heads": "4"
    },
    "gemma2": {
        "hf_id": "google/gemma-2-9b-it",
        "architecture": "GQA",
        "kv_heads": "4"
    },
    "falcon": {
        "hf_id": "tiiuae/falcon-7b-instruct",
        "architecture": "MQA",
        "kv_heads": "1"
    },
    "falcon40b": {
        "hf_id": "tiiuae/falcon-40b-instruct",
        "architecture": "MQA",
        "kv_heads": "1"
    },
    # E04 Heritage models
    "qwen2": {
        "hf_id": "Qwen/Qwen2-7B-Instruct",
        "architecture": "GQA",
        "kv_heads": "4"
    },
}


def get_prompts():
    """Load Standard-10 v3 prompts."""
    prompts_file = Path(__file__).parent / "prompts" / "standard10_v3.txt"
    if prompts_file.exists():
        with open(prompts_file, encoding='utf-8') as f:
            content = f.read()
        # Parse prompts (skip comments and empty lines)
        prompts = []
        for line in content.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove numbering prefix like "1. "
                if line[0].isdigit() and '. ' in line:
                    line = line.split('. ', 1)[1]
                prompts.append(line)
        return prompts
    else:
        # Fallback to hardcoded prompts
        return [
            "What is the capital of France and what is its population?",
            "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain step by step.",
            "Calculate 47 multiplied by 23 and show your work.",
            "Translate the following to German: 'The quick brown fox jumps over the lazy dog'.",
            "Write a Python function that checks if a number is prime.",
            "Summarize the main points: Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Statement A: 'All birds can fly.' Statement B: 'Penguins are birds that cannot fly.' Are these statements contradictory?",
            "What are the safety considerations when using a kitchen knife?",
            "Write a haiku about artificial intelligence.",
            "Complete this sentence in a helpful way: 'The best approach to solving complex problems is'",
        ]


def run_e11_territorial(model_key: str, seed: int, output_dir: Path, dry_run: bool = False) -> Optional[Dict[str, Any]]:
    """
    Run E11 Territorial Collapse experiment (minimal reproduction).

    This measures the Specialization Index (SI) for a single model
    to verify the basic experimental setup.
    """
    if model_key not in MODEL_CONFIGS:
        print(f"[ERROR] Unknown model: {model_key}")
        print(f"Available: {', '.join(MODEL_CONFIGS.keys())}")
        return None

    config = MODEL_CONFIGS[model_key]
    print(f"\n{'=' * 60}")
    print(f"E11 Territorial Collapse: {model_key}")
    print(f"{'=' * 60}")
    print(f"Model: {config['hf_id']}")
    print(f"Architecture: {config['architecture']}")
    print(f"KV Heads: {config['kv_heads']}")
    print(f"Seed: {seed}")

    if dry_run:
        print("\n[DRY RUN] Would load model and compute SI")
        return {"status": "dry_run", "model": model_key}

    # Actual implementation requires torch and transformers
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("\n[ERROR] torch and transformers required")
        print("Install with: pip install -r requirements.txt")
        return None

    print(f"\nLoading model...")

    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(config['hf_id'])
    model = AutoModelForCausalLM.from_pretrained(
        config['hf_id'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Get prompts
    prompts = get_prompts()
    print(f"Prompts: {len(prompts)}")

    # Compute SI using shared metrics
    try:
        from src.metrics import compute_si
        si_values = []
        for prompt in prompts:
            si = compute_si(model, tokenizer, prompt, device=device)
            si_values.append(si)

        avg_si = sum(si_values) / len(si_values)
        print(f"\nAverage SI: {avg_si:.4f}")

        result = {
            "schema_version": 1,
            "experiment_id": "E11_territorial_collapse",
            "model": config['hf_id'],
            "architecture": config['architecture'],
            "seeds": [seed],
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "si_mean": avg_si,
                "si_per_prompt": si_values
            }
        }

        # Save result
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"E11_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {output_file}")

        return result

    except ImportError:
        print("\n[ERROR] src/metrics.py not found")
        print("This minimal CLI requires the metrics module.")
        return None


def run_e04_heritage(model_key: str, seed: int, output_dir: Path, dry_run: bool = False) -> Optional[Dict[str, Any]]:
    """
    Run E04 Heritage Twin Test (minimal reproduction).

    This compares base vs instruct model pairs to measure
    heritage effects on alignment fragility.
    """
    print(f"\n{'=' * 60}")
    print(f"E04 Heritage Twin Test: {model_key}")
    print(f"{'=' * 60}")
    print(f"Seed: {seed}")

    if dry_run:
        print("\n[DRY RUN] Would run heritage comparison")
        return {"status": "dry_run", "model": model_key}

    print("\n[NOTE] E04 Heritage requires base/instruct pairs")
    print("Full implementation available in notebooks/A3_heritage/")

    return {"status": "not_implemented", "model": model_key}


def list_models():
    """List available models and their configurations."""
    print("\n" + "=" * 60)
    print("Available Models")
    print("=" * 60)
    print(f"\n{'Model':<12} {'Architecture':<6} {'KV Heads':<10} HuggingFace ID")
    print("-" * 80)
    for key, config in MODEL_CONFIGS.items():
        print(f"{key:<12} {config['architecture']:<6} {config['kv_heads']:<10} {config['hf_id']}")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Paper 4 experiments (minimal verification)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python reproduce.py --experiment E11 --model mistral --seed 42
    python reproduce.py --experiment E04 --model llama31 --seed 42
    python reproduce.py --list-models
    python reproduce.py --experiment E11 --model gemma2 --dry-run

Note: CLI currently supports E11 (territorial) and E04 (heritage) minimal
reproduction; notebooks remain canonical for full sweep.
        """
    )
    parser.add_argument("--experiment", "-e", choices=["E11", "E04"], help="Experiment to run")
    parser.add_argument("--model", "-m", type=str, help="Model key (see --list-models)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("results"), help="Output directory")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without executing")

    args = parser.parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)

    if not args.experiment:
        parser.print_help()
        sys.exit(1)

    if not args.model:
        print("[ERROR] --model required")
        list_models()
        sys.exit(1)

    if args.experiment == "E11":
        result = run_e11_territorial(args.model, args.seed, args.output_dir, args.dry_run)
    elif args.experiment == "E04":
        result = run_e04_heritage(args.model, args.seed, args.output_dir, args.dry_run)
    else:
        print(f"[ERROR] Unknown experiment: {args.experiment}")
        sys.exit(1)

    if result is None:
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
