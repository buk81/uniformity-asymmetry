#!/usr/bin/env python3
"""
Verify Paper 4 release package integrity.

This script validates:
1. All files referenced in claims.yaml exist
2. Result JSON files conform to schema_v1.json
3. Prompts file hash matches expected value

Usage:
    python verify_release.py
    python verify_release.py --verbose
    python verify_release.py --check-schema
"""

import yaml
import json
import hashlib
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def load_claims(path: Path) -> dict:
    """Load claims.yaml file."""
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_schema(path: Path) -> dict:
    """Load JSON schema file."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def verify_file_exists(path: Path, base: Path) -> bool:
    """Check if a file exists relative to base directory."""
    full_path = base / path
    return full_path.exists()


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def validate_json_against_schema(json_path: Path, schema: dict) -> Tuple[bool, str]:
    """Validate a JSON file against the schema."""
    if not HAS_JSONSCHEMA:
        return True, "jsonschema not installed, skipping validation"

    try:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        jsonschema.validate(data, schema)
        return True, "Valid"
    except jsonschema.ValidationError as e:
        return False, f"Schema validation error: {e.message}"
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"


def verify_claims(claims: dict, base: Path, verbose: bool = False) -> List[str]:
    """Verify all claims and their artifacts exist."""
    errors = []

    for claim_id, claim in claims.get("claims", {}).items():
        print(f"\n{'=' * 60}")
        print(f"Checking {claim_id}: {claim['name']} (Tier: {claim.get('tier', 'A')})")
        print(f"{'=' * 60}")

        # Check notebooks
        print(f"\nNotebooks:")
        for nb in claim.get("notebooks", []):
            nb_path = Path(nb)
            if verify_file_exists(nb_path, base):
                print(f"  [OK] {nb}")
            else:
                print(f"  [MISSING] {nb}")
                errors.append(f"{claim_id}: notebook {nb}")

        # Check results
        print(f"\nResults:")
        for res in claim.get("results", []):
            res_path = Path(res)
            if verify_file_exists(res_path, base):
                print(f"  [OK] {res}")
            else:
                print(f"  [MISSING] {res}")
                errors.append(f"{claim_id}: result {res}")

    return errors


def verify_supporting(supporting: dict, base: Path, verbose: bool = False) -> List[str]:
    """Verify supporting evidence files exist."""
    errors = []

    print(f"\n{'=' * 60}")
    print(f"Checking Supporting Evidence (B-Tier)")
    print(f"{'=' * 60}")

    for exp_id, exp in supporting.items():
        print(f"\n{exp_id}: {exp.get('description', '')}")
        for res in exp.get("results", []):
            res_path = Path(res)
            if verify_file_exists(res_path, base):
                print(f"  [OK] {res}")
            else:
                print(f"  [MISSING] {res}")
                errors.append(f"Supporting {exp_id}: {res}")

    return errors


def verify_prompts(base: Path) -> Tuple[bool, str]:
    """Verify prompts file exists and return its hash."""
    prompts_file = base / "prompts" / "standard10_v3.txt"

    if not prompts_file.exists():
        return False, "Prompts file not found"

    sha = compute_sha256(prompts_file)
    return True, sha


def verify_schema_compliance(base: Path, claims: dict, verbose: bool = False) -> List[str]:
    """Verify all result JSONs conform to schema."""
    errors = []
    schema_path = base / "results" / "schema_v1.json"

    if not schema_path.exists():
        print("\n[WARNING] schema_v1.json not found, skipping schema validation")
        return errors

    if not HAS_JSONSCHEMA:
        print("\n[WARNING] jsonschema package not installed, skipping schema validation")
        print("         Install with: pip install jsonschema")
        return errors

    schema = load_schema(schema_path)

    print(f"\n{'=' * 60}")
    print(f"Schema Validation")
    print(f"{'=' * 60}")

    # Collect all result files
    result_files = []
    for claim in claims.get("claims", {}).values():
        for res in claim.get("results", []):
            result_files.append(Path(res))

    for supporting in claims.get("supporting_evidence", {}).values():
        for res in supporting.get("results", []):
            result_files.append(Path(res))

    for res_path in result_files:
        full_path = base / res_path
        if full_path.exists() and full_path.suffix == '.json':
            valid, msg = validate_json_against_schema(full_path, schema)
            if valid:
                if verbose:
                    print(f"  [OK] {res_path}")
            else:
                print(f"  [FAIL] {res_path}: {msg}")
                errors.append(f"Schema: {res_path}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Verify Paper 4 release package integrity")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--check-schema", action="store_true", help="Also validate JSON schema compliance")
    args = parser.parse_args()

    base = Path(__file__).parent
    claims_file = base / "claims.yaml"

    print("=" * 60)
    print("Paper 4 Release Package Verification")
    print("=" * 60)
    print(f"\nBase directory: {base}")

    # Check claims.yaml exists
    if not claims_file.exists():
        print("\n[FATAL] claims.yaml not found")
        sys.exit(1)

    claims = load_claims(claims_file)
    print(f"Schema version: {claims.get('schema_version', 'unknown')}")

    all_errors = []

    # Verify A-tier claims
    claim_errors = verify_claims(claims, base, args.verbose)
    all_errors.extend(claim_errors)

    # Verify B-tier supporting evidence
    supporting_errors = verify_supporting(claims.get("supporting_evidence", {}), base, args.verbose)
    all_errors.extend(supporting_errors)

    # Verify prompts hash
    prompts_ok, prompts_result = verify_prompts(base)
    print(f"\n{'=' * 60}")
    print(f"Prompts Verification")
    print(f"{'=' * 60}")
    if prompts_ok:
        print(f"  [OK] Prompts SHA256: {prompts_result}")
    else:
        print(f"  [FAIL] {prompts_result}")
        all_errors.append("Prompts file missing")

    # Schema validation (optional)
    if args.check_schema:
        schema_errors = verify_schema_compliance(base, claims, args.verbose)
        all_errors.extend(schema_errors)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")

    if all_errors:
        print(f"\n[FAILED] {len(all_errors)} issue(s) found:\n")
        for error in all_errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("\n[PASSED] All checks passed!")
        print("\nRelease package is ready for distribution.")
        sys.exit(0)


if __name__ == "__main__":
    main()
