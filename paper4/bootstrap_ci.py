#!/usr/bin/env python3
"""
Bootstrap Confidence Interval Calculator for Paper 4 Experiments

Handles:
- E04: Heritage/Twin tests (repetition-score metric)
- E08: Alignment Density (delta_si across model families)
- E11: State-Dependency/Indra (SI under noise injection)

Statistical Methods:
1. Percentile Bootstrap (basic)
2. BCa Bootstrap (bias-corrected and accelerated) for small samples
3. Seed-level bootstrap (n=3 seeds)
4. Effect size (Cohen's d) with CI

Usage:
    python bootstrap_ci.py results/E04_qwen_twin_*.json
    python bootstrap_ci.py results/E11_indra_*.json
    python bootstrap_ci.py results/E08c_*.json
    python bootstrap_ci.py --all results/

Author: Paper 4 Statistical Hardening
Date: 2026-01-13
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import warnings

# Suppress warnings for small sample sizes
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# Configuration
# =============================================================================

N_BOOTSTRAP = 10000  # Bootstrap iterations
CI_LEVEL = 0.95  # 95% confidence interval
SEEDS = [42, 123, 456]  # Standard seeds


@dataclass
class BootstrapResult:
    """Container for bootstrap CI results."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_samples: int
    method: str
    effect_size_d: Optional[float] = None  # Cohen's d

    def __repr__(self):
        return (f"{self.point_estimate:.4f} "
                f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
                f"(SE={self.std_error:.4f}, n={self.n_samples}, {self.method})")

    def to_dict(self) -> Dict:
        return {
            'point_estimate': self.point_estimate,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'std_error': self.std_error,
            'n_samples': self.n_samples,
            'method': self.method,
            'effect_size_d': self.effect_size_d,
            'ci_width': self.ci_upper - self.ci_lower,
            'significant': not (self.ci_lower <= 0 <= self.ci_upper)
        }


# =============================================================================
# Core Bootstrap Functions
# =============================================================================

def percentile_bootstrap(data: np.ndarray, n_boot: int = N_BOOTSTRAP,
                         ci_level: float = CI_LEVEL) -> BootstrapResult:
    """
    Basic percentile bootstrap for small samples.

    Args:
        data: 1D array of observations (e.g., 3 seed values)
        n_boot: Number of bootstrap iterations
        ci_level: Confidence level (0.95 = 95% CI)

    Returns:
        BootstrapResult with CI bounds
    """
    data = np.asarray(data)
    n = len(data)

    if n < 2:
        return BootstrapResult(
            point_estimate=float(data[0]) if n == 1 else np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            std_error=np.nan,
            n_samples=n,
            method='single_point'
        )

    # Bootstrap resampling
    boot_means = np.zeros(n_boot)
    rng = np.random.default_rng(42)

    for i in range(n_boot):
        boot_sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(boot_sample)

    # Percentile CI
    alpha = (1 - ci_level) / 2
    ci_lower = np.percentile(boot_means, alpha * 100)
    ci_upper = np.percentile(boot_means, (1 - alpha) * 100)

    return BootstrapResult(
        point_estimate=np.mean(data),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=np.std(boot_means),
        n_samples=n,
        method='percentile'
    )


def bca_bootstrap(data: np.ndarray, n_boot: int = N_BOOTSTRAP,
                  ci_level: float = CI_LEVEL) -> BootstrapResult:
    """
    BCa (Bias-Corrected and Accelerated) bootstrap.
    Better for small samples and skewed distributions.

    Args:
        data: 1D array of observations
        n_boot: Number of bootstrap iterations
        ci_level: Confidence level

    Returns:
        BootstrapResult with BCa-corrected CI bounds
    """
    data = np.asarray(data)
    n = len(data)

    if n < 3:
        # Fall back to percentile for very small samples
        return percentile_bootstrap(data, n_boot, ci_level)

    # Bootstrap resampling
    boot_means = np.zeros(n_boot)
    rng = np.random.default_rng(42)

    for i in range(n_boot):
        boot_sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(boot_sample)

    # Bias correction factor
    theta_hat = np.mean(data)
    z0 = stats.norm.ppf(np.mean(boot_means < theta_hat))

    # Acceleration factor (jackknife)
    jackknife_means = np.zeros(n)
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        jackknife_means[i] = np.mean(jackknife_sample)

    theta_dot = np.mean(jackknife_means)
    num = np.sum((theta_dot - jackknife_means) ** 3)
    denom = 6 * (np.sum((theta_dot - jackknife_means) ** 2) ** 1.5)

    a = num / denom if denom != 0 else 0

    # BCa adjusted percentiles
    alpha = (1 - ci_level) / 2
    z_alpha_lower = stats.norm.ppf(alpha)
    z_alpha_upper = stats.norm.ppf(1 - alpha)

    # Adjusted quantiles
    if not np.isnan(z0) and not np.isinf(z0):
        alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        alpha2 = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))
    else:
        alpha1, alpha2 = alpha, 1 - alpha

    # Clamp to valid range
    alpha1 = np.clip(alpha1, 0.001, 0.999)
    alpha2 = np.clip(alpha2, 0.001, 0.999)

    ci_lower = np.percentile(boot_means, alpha1 * 100)
    ci_upper = np.percentile(boot_means, alpha2 * 100)

    return BootstrapResult(
        point_estimate=theta_hat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=np.std(boot_means),
        n_samples=n,
        method='BCa'
    )


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    if pooled_std == 0:
        return np.nan

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def paired_bootstrap_diff(data1: np.ndarray, data2: np.ndarray,
                          n_boot: int = N_BOOTSTRAP,
                          ci_level: float = CI_LEVEL) -> BootstrapResult:
    """
    Bootstrap CI for paired differences (e.g., base vs instruct).

    Args:
        data1: First condition (e.g., base model SI per seed)
        data2: Second condition (e.g., instruct model SI per seed)

    Returns:
        BootstrapResult for the difference
    """
    diff = np.asarray(data1) - np.asarray(data2)
    result = bca_bootstrap(diff, n_boot, ci_level)
    result.effect_size_d = cohens_d(np.asarray(data1), np.asarray(data2))
    return result


# =============================================================================
# E04 Heritage/Twin Experiment Analysis
# =============================================================================

def analyze_e04_results(results: Dict) -> Dict[str, Any]:
    """
    Analyze E04 Heritage/Twin experiment with Bootstrap CI.

    E04 measures repetition-score fragility across noise levels.
    Key metric: fragility (slope of repetition vs noise curve)
    """
    output = {
        'experiment': results.get('experiment', 'E04'),
        'model_config': results.get('model_config', {}),
        'bootstrap_ci': {}
    }

    seed_results = results.get('results', {})

    for model_type in ['base', 'instruct']:
        if model_type not in seed_results:
            continue

        model_data = seed_results[model_type].get('seed_results', {})
        if not model_data:
            continue

        output['bootstrap_ci'][model_type] = {}

        # Collect fragility values per layer range
        for layer_range in ['early', 'middle', 'late', 'all']:
            fragilities = []

            for seed in SEEDS:
                seed_key = str(seed)
                if seed_key in model_data and layer_range in model_data[seed_key]:
                    frag = model_data[seed_key][layer_range].get('fragility')
                    if frag is not None:
                        fragilities.append(frag)

            if len(fragilities) >= 2:
                result = bca_bootstrap(np.array(fragilities))
                output['bootstrap_ci'][model_type][layer_range] = result.to_dict()

    # Heritage verdict with CI
    if 'base' in output['bootstrap_ci'] and 'instruct' in output['bootstrap_ci']:
        output['heritage_comparison'] = {}

        for layer_range in ['early', 'middle', 'late', 'all']:
            base_frags = []
            inst_frags = []

            for seed in SEEDS:
                seed_key = str(seed)
                base_data = seed_results.get('base', {}).get('seed_results', {})
                inst_data = seed_results.get('instruct', {}).get('seed_results', {})

                if seed_key in base_data and layer_range in base_data[seed_key]:
                    f = base_data[seed_key][layer_range].get('fragility')
                    if f is not None:
                        base_frags.append(f)

                if seed_key in inst_data and layer_range in inst_data[seed_key]:
                    f = inst_data[seed_key][layer_range].get('fragility')
                    if f is not None:
                        inst_frags.append(f)

            if len(base_frags) >= 2 and len(inst_frags) >= 2:
                diff_result = paired_bootstrap_diff(
                    np.array(inst_frags),
                    np.array(base_frags)
                )
                output['heritage_comparison'][layer_range] = {
                    'delta_fragility': diff_result.to_dict(),
                    'verdict': 'PROTECTED' if diff_result.ci_upper < 0.05 else
                              ('DAMAGED' if diff_result.ci_lower > 0.15 else 'UNCERTAIN'),
                    'interpretation': (
                        'Instruct less fragile than Base' if diff_result.point_estimate < 0
                        else 'Instruct more fragile than Base'
                    )
                }

    return output


# =============================================================================
# E11 Indra/State-Dependency Analysis
# =============================================================================

def analyze_e11_results(results: Dict) -> Dict[str, Any]:
    """
    Analyze E11 Indra/State-Dependency experiment with Bootstrap CI.

    Handles two formats:
    1. Old format: single seed with treatments array
    2. E11T format: multi-seed with all_seed_results containing global/local

    E11 measures SI change under noise injection.
    Key metric: SI delta at different noise levels
    """
    output = {
        'experiment': results.get('experiment', 'E11'),
        'model': results.get('model', ''),
        'architecture': results.get('architecture', ''),
        'rho': results.get('head_density_rho', None),
        'reference': results.get('reference', {}),
        'bootstrap_ci': {}
    }

    # Try E11T format first (multi-seed with base/instruct structure)
    if 'results' in results and isinstance(results['results'], dict):
        res = results['results']

        # Check for E11T format: has 'base' or 'instruct' with 'all_seed_results'
        for model_type in ['base', 'instruct']:
            if model_type not in res:
                continue

            model_data = res[model_type]
            all_seed_results = model_data.get('all_seed_results', {})

            if all_seed_results:
                output['bootstrap_ci'][model_type] = {}

                # Collect data across seeds for each region and noise level
                # Structure: all_seed_results[seed]['global'][region_idx]['tests'][noise_idx]
                region_data = {}  # region -> noise_level -> [si_values across seeds]

                for seed in SEEDS:
                    seed_key = str(seed)
                    if seed_key not in all_seed_results:
                        continue

                    seed_data = all_seed_results[seed_key]

                    # Handle 'global' measurements
                    global_tests = seed_data.get('global', [])
                    for region_entry in global_tests:
                        region = region_entry.get('region', 'unknown')
                        if region not in region_data:
                            region_data[region] = {'si': {}, 'change_pct': {}}

                        for test in region_entry.get('tests', []):
                            noise = test.get('noise', 0)
                            si = test.get('si', 0)
                            change = test.get('change_pct', 0)

                            if noise not in region_data[region]['si']:
                                region_data[region]['si'][noise] = []
                                region_data[region]['change_pct'][noise] = []

                            region_data[region]['si'][noise].append(si)
                            region_data[region]['change_pct'][noise].append(change)

                # Compute CI for each region
                for region, data in region_data.items():
                    output['bootstrap_ci'][model_type][region] = {}

                    # CI for SI at key noise levels
                    for noise_level in [0.0, 0.1, 0.2]:
                        if noise_level in data['si'] and len(data['si'][noise_level]) >= 2:
                            si_values = np.array(data['si'][noise_level])
                            change_values = np.array(data['change_pct'][noise_level])

                            result_si = bca_bootstrap(si_values)
                            result_change = bca_bootstrap(change_values)

                            output['bootstrap_ci'][model_type][region][f'si_noise_{noise_level}'] = {
                                **result_si.to_dict(),
                                'metric': 'absolute_si'
                            }
                            output['bootstrap_ci'][model_type][region][f'change_noise_{noise_level}'] = {
                                **result_change.to_dict(),
                                'metric': 'change_pct'
                            }

                # Add baseline SI if available
                baseline_global = model_data.get('baseline_global', {})
                if baseline_global:
                    output['bootstrap_ci'][model_type]['baseline'] = {
                        'si': baseline_global.get('si', baseline_global.get('specialization_index')),
                        'corr': baseline_global.get('mean_corr', baseline_global.get('mean_head_correlation'))
                    }

        # If we got E11T data, return early
        if any(mt in output['bootstrap_ci'] for mt in ['base', 'instruct']):
            # Compare base vs instruct if both exist
            if 'base' in output['bootstrap_ci'] and 'instruct' in output['bootstrap_ci']:
                output['comparison'] = {}

                base_baseline = output['bootstrap_ci']['base'].get('baseline', {})
                inst_baseline = output['bootstrap_ci']['instruct'].get('baseline', {})

                if base_baseline.get('si') is not None and inst_baseline.get('si') is not None:
                    delta_si = inst_baseline['si'] - base_baseline['si']
                    output['comparison']['delta_si_absolute'] = delta_si
                    output['comparison']['interpretation'] = (
                        'HEAL (Instruct more specialized)' if delta_si > 0
                        else 'HARM (Instruct less specialized)'
                    )

            return output

    # Try nested format: results[model_type][treatments]
    if 'results' in results and isinstance(results['results'], dict):
        res = results['results']
        for model_type in ['base', 'instruct']:
            if model_type in res and 'treatments' in res[model_type]:
                model_data = res[model_type]
                treatments = model_data.get('treatments', [])

                if treatments:
                    output['bootstrap_ci'][model_type] = {}

                    for treatment in treatments:
                        region = treatment.get('region', 'unknown')
                        output['bootstrap_ci'][model_type][region] = {}

                        noise_tests = treatment.get('noise_tests', [])
                        # Single-seed: can't compute CI, just report values
                        for test in noise_tests:
                            noise_std = test.get('noise_std', 0)
                            si = test.get('specialization_index', test.get('si', 0))
                            delta = test.get('si_delta', test.get('si_delta_pct', 0))

                            output['bootstrap_ci'][model_type][region][f'noise_{noise_std}'] = {
                                'point_estimate': si,
                                'delta': delta,
                                'n_samples': 1,
                                'method': 'single_seed_no_ci'
                            }

        if any(mt in output['bootstrap_ci'] for mt in ['base', 'instruct']):
            return output

    # Fall back to old format: single seed with treatments array
    treatments = results.get('results', {}).get('treatments', [])
    if not treatments:
        treatments = results.get('treatments', [])

    # Group by region and noise level
    region_data = {}

    for treatment in treatments:
        region = treatment.get('region', 'unknown')
        if region not in region_data:
            region_data[region] = {'noise_levels': {}}

        noise_tests = treatment.get('noise_tests', [])
        for test in noise_tests:
            noise_std = test.get('noise_std', 0)
            si_delta = test.get('si_delta', test.get('change_pct', 0))

            if noise_std not in region_data[region]['noise_levels']:
                region_data[region]['noise_levels'][noise_std] = []

            region_data[region]['noise_levels'][noise_std].append(si_delta)

    # Compute CI for each region and noise level
    for region, data in region_data.items():
        output['bootstrap_ci'][region] = {}

        for noise_std, deltas in data['noise_levels'].items():
            if len(deltas) >= 2:
                result = bca_bootstrap(np.array(deltas))
                output['bootstrap_ci'][region][f'noise_{noise_std}'] = result.to_dict()

        # Key metric: SI change at noise=0.1 (standard test level)
        if 0.1 in data['noise_levels'] and len(data['noise_levels'][0.1]) >= 2:
            result = bca_bootstrap(np.array(data['noise_levels'][0.1]))
            output['bootstrap_ci'][region]['key_metric_noise_0.1'] = {
                **result.to_dict(),
                'interpretation': (
                    'HEAL (SI increases)' if result.ci_lower > 0
                    else ('HARM (SI decreases)' if result.ci_upper < 0
                          else 'UNCERTAIN (CI crosses zero)')
                )
            }

    return output


# =============================================================================
# E08 Alignment Density Analysis
# =============================================================================

def analyze_e08_results(results: Dict) -> Dict[str, Any]:
    """
    Analyze E08 Alignment Density experiment with Bootstrap CI.

    E08 measures delta_si across model families and sizes.
    Key metric: ρ-SI correlation, sign flips
    """
    output = {
        'experiment': results.get('experiment', 'E08'),
        'rho_crit_reference': results.get('rho_crit_reference', 0.267),
        'bootstrap_ci': {},
        'family_analysis': {}
    }

    family_results = results.get('results', [])

    # Group by family
    families = {}
    for result in family_results:
        family = result.get('family', 'unknown')
        if family not in families:
            families[family] = {'sizes': [], 'rhos': [], 'delta_sis': []}

        # Get seed results
        seed_results = result.get('seed_results', [result])

        for sr in seed_results:
            if sr.get('status') == 'error':
                continue

            size = sr.get('size', result.get('size', ''))
            rho = sr.get('rho', result.get('rho_head', None))

            # Extract delta_si
            delta_si = sr.get('delta_si_pct')
            if delta_si is None:
                base_si = sr.get('base_si', 0)
                inst_si = sr.get('instruct_si', 0)
                if base_si != 0:
                    delta_si = ((inst_si - base_si) / base_si) * 100

            if rho is not None and delta_si is not None:
                families[family]['sizes'].append(size)
                families[family]['rhos'].append(rho)
                families[family]['delta_sis'].append(delta_si)

    # Analyze each family
    for family, data in families.items():
        if len(data['delta_sis']) < 2:
            continue

        delta_si_array = np.array(data['delta_sis'])
        rho_array = np.array(data['rhos'])

        # Bootstrap CI on delta_si
        result = bca_bootstrap(delta_si_array)
        output['bootstrap_ci'][family] = {
            'delta_si': result.to_dict(),
            'n_measurements': len(delta_si_array)
        }

        # ρ-SI correlation (if enough variation in ρ)
        if len(np.unique(rho_array)) > 1 and len(rho_array) >= 3:
            try:
                r, p = stats.pearsonr(rho_array, delta_si_array)
                output['family_analysis'][family] = {
                    'rho_si_correlation': r,
                    'p_value': p,
                    'significant': p < 0.05,
                    'interpretation': (
                        'Higher ρ → Lower ΔSI (expected)' if r < 0
                        else 'Higher ρ → Higher ΔSI (unexpected)'
                    )
                }
            except Exception:
                pass

    return output


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def detect_experiment_type(results: Dict) -> str:
    """Auto-detect experiment type from JSON structure."""
    exp = results.get('experiment', '').upper()

    if 'E04' in exp or 'TWIN' in exp or 'HERITAGE' in exp:
        return 'E04'
    elif 'E11' in exp or 'INDRA' in exp or 'STATE' in exp:
        return 'E11'
    elif 'E08' in exp or 'DENSITY' in exp or 'ALIGNMENT' in exp:
        return 'E08'

    # Fallback: check structure
    if 'seed_results' in results.get('results', {}).get('base', {}):
        return 'E04'
    elif 'treatments' in results.get('results', {}):
        return 'E11'
    elif 'family' in str(results.get('results', [])):
        return 'E08'

    return 'unknown'


def analyze_results_file(filepath: Path) -> Dict[str, Any]:
    """Load and analyze a single results file."""
    with open(filepath, 'r') as f:
        results = json.load(f)

    exp_type = detect_experiment_type(results)

    if exp_type == 'E04':
        return analyze_e04_results(results)
    elif exp_type == 'E11':
        return analyze_e11_results(results)
    elif exp_type == 'E08':
        return analyze_e08_results(results)
    else:
        return {'error': f'Unknown experiment type for {filepath.name}', 'raw': results}


def generate_summary_table(analyses: List[Dict]) -> str:
    """Generate markdown summary table of all bootstrap results."""
    lines = [
        "# Bootstrap CI Summary\n",
        "| Experiment | Metric | Point Est. | 95% CI | Significant | n |",
        "|------------|--------|------------|--------|-------------|---|"
    ]

    def add_result_line(exp: str, metric: str, result: Dict):
        """Add a single result line to the table."""
        pe = result.get('point_estimate', result.get('delta', 0))
        method = result.get('method', '')
        n = result.get('n_samples', 1)

        if method == 'single_seed_no_ci':
            # Single seed - no CI
            delta = result.get('delta', 0)
            lines.append(
                f"| {exp} | {metric} | {pe:.4f} | (δ={delta:.4f}) | n/a | {n} |"
            )
        elif 'ci_lower' in result and 'ci_upper' in result:
            # Has CI
            ci_l = result['ci_lower']
            ci_u = result['ci_upper']
            sig = '✅' if result.get('significant', False) else '❌'
            lines.append(
                f"| {exp} | {metric} | {pe:.4f} | [{ci_l:.4f}, {ci_u:.4f}] | {sig} | {n} |"
            )
        else:
            # Other format
            lines.append(
                f"| {exp} | {metric} | {pe:.4f} | - | - | {n} |"
            )

    for analysis in analyses:
        exp = analysis.get('experiment', 'Unknown')
        bootstrap_ci = analysis.get('bootstrap_ci', {})

        def process_dict(prefix: str, d: Dict, depth: int = 0):
            """Recursively process nested dictionaries."""
            if depth > 3:  # Prevent infinite recursion
                return

            for key, value in d.items():
                if not isinstance(value, dict):
                    continue

                metric = f"{prefix}/{key}" if prefix else key

                # Check if this is a result dict
                if 'point_estimate' in value or 'delta' in value or 'method' in value:
                    add_result_line(exp, metric, value)
                else:
                    # Recurse into nested structure
                    process_dict(metric, value, depth + 1)

        process_dict('', bootstrap_ci)

    return '\n'.join(lines)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Bootstrap CI Calculator for Paper 4')
    parser.add_argument('files', nargs='+', help='JSON result files or directory with --all')
    parser.add_argument('--all', action='store_true', help='Process all JSON files in directory')
    parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    parser.add_argument('--summary', '-s', action='store_true', help='Print markdown summary')
    args = parser.parse_args()

    files_to_process = []

    for path_str in args.files:
        path = Path(path_str)
        if path.is_dir() and args.all:
            files_to_process.extend(path.glob('*.json'))
        elif path.is_file() and path.suffix == '.json':
            files_to_process.append(path)
        elif '*' in path_str:
            files_to_process.extend(Path('.').glob(path_str))

    if not files_to_process:
        print("No JSON files found to process")
        return

    print(f"Processing {len(files_to_process)} files...")

    all_analyses = []
    for filepath in sorted(files_to_process):
        print(f"  Analyzing: {filepath.name}")
        try:
            analysis = analyze_results_file(filepath)
            analysis['source_file'] = str(filepath)
            all_analyses.append(analysis)
        except Exception as e:
            print(f"    ERROR: {e}")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(all_analyses, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    if args.summary:
        print("\n" + generate_summary_table(all_analyses))
    else:
        # Print brief summary
        print(f"\nProcessed {len(all_analyses)} experiments")
        for analysis in all_analyses:
            exp = analysis.get('experiment', 'Unknown')
            n_ci = len(analysis.get('bootstrap_ci', {}))
            print(f"  {exp}: {n_ci} CI computed")


if __name__ == '__main__':
    main()
